import gc
import os
import pickle
import uuid

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tenacity import retry, stop_after_attempt, wait_fixed


def raw_match_counts(data):
    """每期与前 lag 期（更旧）相同数字的个数，形状 (N, 3)。

    时间序列约定：data 按时间升序排列（旧 -> 新），索引越小越旧。
    """
    counts = np.zeros((len(data), 3), dtype=np.float32)
    for lag in (1, 2, 3):
        counts[lag:, lag - 1] = (data[lag:] == data[:-lag]).sum(axis=1)
    return counts


# load_data 未显式指定时的默认期数（GUI 样本数量输入框同值）。
DEFAULT_NUM_ROWS = 7000
# 单次「更新数据」最多请求的接口页数（PAGE_SIZE=100 → 最多 3000 期）。
MAX_API_PAGES = 30


class DataProcessor:
    def __init__(self, file_path, fetch_url):
        self.file_path = str(file_path)
        self.fetch_url = fetch_url
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.match_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None             # (N, 3) int32，时间升序：旧 -> 新
        self.features = None         # (N, 3) float32
        self.recent_samples = None   # 新 -> 旧，仅用于界面展示
        self.last_period = None
        self._last_num_rows = DEFAULT_NUM_ROWS
        # build_feature_matrix(fit_scalers=False) 的全量缓存；数据或 scaler
        # 变化（加载/拟合）时置 None。特征逐行独立，upto 切片可直接取前缀。
        self._matrix_cache = None

    def load_data(self, num_rows, log_callback):
        """读取 Excel，并统一按时间升序（旧 -> 新）组织。"""
        if not os.path.exists(self.file_path):
            log_callback(f"文件 {self.file_path} 未找到")
            return False
        try:
            df = pd.read_excel(self.file_path, engine='openpyxl')
            self._apply_dataframe(df, num_rows, log_callback)
            return True
        except Exception as e:
            log_callback(f"加载数据失败：{str(e)}")
            return False
        finally:
            gc.collect()

    def _apply_dataframe(self, df, num_rows, log_callback):
        """解析已读入的 DataFrame（load_data 与 fetch_and_update 共用）。"""
        if 'Numbers' not in df.columns or 'Period' not in df.columns:
            raise ValueError("Excel缺少必要列：['Numbers', 'Period']")

        # 取最近 num_rows 期并翻转为时间升序；不再伪造开奖号码做数据增强。
        df = df.sort_values(by='Period', ascending=True).tail(num_rows).reset_index(drop=True)
        log_callback(f"加载了 {len(df)} 条数据")

        numbers = [str(row).split() for row in df['Numbers']]
        if any(len(row) != 3 or any(len(t) != 1 or not t.isdigit() for t in row)
               for row in numbers):
            raise ValueError("Numbers 列必须是由空格分隔的 3 个 0-9 数字")

        self.data = np.array([[int(t) for t in row] for row in numbers], dtype=np.int32)
        self.features = np.array([self._calculate_features(row) for row in self.data],
                                 dtype=np.float32)

        self.last_period = int(df['Period'].iloc[-1])
        self.recent_samples = df.tail(50).iloc[::-1].reset_index(drop=True).copy()
        self._last_num_rows = num_rows
        self._matrix_cache = None
        self._enhance_recent_samples()
        log_callback(f"数据按时间升序组织，最新期号 {self.last_period}，共 {len(self.data)} 期")

    def build_feature_matrix(self, fit_scalers=True, upto=None):
        """构建每期 9 维特征：3 个归一化号码 + 3 个 lag 匹配数 + 3 个统计特征。

        训练与预测共用此函数，保证特征定义完全一致。
        fit_scalers=True 用于训练；加载已保存状态后预测时传 False。
        upto: 只使用前 upto 期（时间升序切片 data[:upto]），供滚动预测做
        严格无未来泄漏的特征组装；None 表示全部数据，行为与旧版一致。

        特征逐行独立（lag 只向前看），scaler 是逐列线性变换，因此
        upto=t 的结果恒等于全量矩阵的前 t 行——全量矩阵只算一次并缓存，
        反复预测/滚动评估直接取前缀切片。
        """
        if self.data is None:
            raise RuntimeError("数据尚未加载")
        end = len(self.data) if upto is None else max(0, min(int(upto), len(self.data)))
        if fit_scalers or self._matrix_cache is None:
            raw = (raw_match_counts(self.data) if len(self.data) > 0
                   else np.zeros((0, 3), dtype=np.float32))
            if fit_scalers:
                # fit 只用前 end 期（与旧版一致）；fit 后再变换全量数据，
                # 保证缓存覆盖完整数据且前 end 行与旧版逐切片结果一致。
                self.scaler.fit(self.data[:end].reshape(-1, 1))
                self.match_scaler.fit(raw[:end])
                self.feature_scaler.fit(self.features[:end])
            norm_data = self.scaler.transform(self.data.reshape(-1, 1)).reshape(-1, 3)
            norm_match = self.match_scaler.transform(raw)
            norm_feat = self.feature_scaler.transform(self.features)
            self._matrix_cache = np.hstack(
                (norm_data, norm_match, norm_feat)).astype(np.float32)
        return self._matrix_cache[:end]

    def _calculate_features(self, numbers):
        odd_count = sum(1 for n in numbers if n % 2 == 1) / 3
        big_count = sum(1 for n in numbers if n >= 5) / 3
        total_sum = sum(numbers)
        return [odd_count, big_count, total_sum]

    def _numbers_to_group(self, numbers):
        return ''.join(map(str, sorted(set(numbers))))

    def _enhance_recent_samples(self):
        parsed = [list(map(int, str(row).split())) for row in self.recent_samples['Numbers']]
        feats = [self._calculate_features(row) for row in parsed]
        # 用 round 而非 int：浮点 1/3*3 可能是 0.9999...，int 会错显示成 0。
        self.recent_samples['Group'] = [self._numbers_to_group(row) for row in parsed]
        self.recent_samples['OddEven'] = [f"{round(f[0] * 3)}:{3 - round(f[0] * 3)}" for f in feats]
        self.recent_samples['BigSmall'] = [f"{round(f[1] * 3)}:{3 - round(f[1] * 3)}" for f in feats]
        self.recent_samples['Sum'] = [f[2] for f in feats]

    # 体彩官方接口单页最多返回 100 条。
    PAGE_SIZE = 100

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_api_page(self, page_no, log_callback=None):
        url = self.fetch_url.format(page_size=self.PAGE_SIZE, page_no=page_no)
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0 Safari/537.36'),
            'Accept': 'application/json',
            'Referer': 'https://static.sporttery.cn/',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if not payload.get('success'):
            raise ValueError(f"接口返回失败：{payload.get('errorMessage', '未知错误')}")
        return payload['value']['list']

    def _fetch_api_data(self, num_periods, last_known, log_callback):
        """从体彩官方 JSON 接口拉取最新开奖。

        排列5（gameNo=350133）的前 3 位即当期排列3 开奖号码，
        已用历史数据逐期对拍验证。接口分页返回（新 -> 旧），
        拉到覆盖本地已有最新期号或达到请求数量为止。
        """
        records = []
        seen = set()
        max_pages = min(MAX_API_PAGES, max(1, -(-int(num_periods) // self.PAGE_SIZE)))
        for page in range(1, max_pages + 1):
            rows = self._fetch_api_page(page, log_callback)
            if not rows:
                break
            page_periods = []
            for item in rows:
                result = str(item.get('lotteryDrawResult', '')).split()
                if len(result) < 3 or not all(t.isdigit() and len(t) == 1
                                              for t in result[:3]):
                    continue
                try:
                    period = int(item['lotteryDrawNum'])
                except (KeyError, TypeError, ValueError):
                    continue
                page_periods.append(period)
                if period in seen:
                    continue
                seen.add(period)
                records.append({'Period': period,
                                'Numbers': ' '.join(result[:3])})
            if not page_periods:
                break
            oldest = min(page_periods)
            log_callback(f"已获取第 {page} 页，累计 {len(records)} 条（最旧期号 {oldest}）")
            if oldest <= last_known or len(records) >= int(num_periods):
                break
        if records:
            oldest_fetched = min(r['Period'] for r in records)
            if oldest_fetched > last_known:
                log_callback(f"警告：已达最大页数（{max_pages} 页）仍未追上本地最新期号 "
                             f"{last_known}，接口数据最早只到 {oldest_fetched}；"
                             "如需更早历史请直接使用本地数据文件")
        return records

    def fetch_and_update(self, num_periods, log_callback):
        """抓取新期号并写回 Excel，随后直接用合并结果刷新内存（不再重读文件）。"""
        log_callback(f"正在从体彩官方接口获取最新开奖数据（最多检查 {num_periods} 期）...")
        try:
            df = pd.read_excel(self.file_path, engine='openpyxl')
            df['Period'] = df['Period'].astype(int)
            last_known = int(df['Period'].max())

            web_data = self._fetch_api_data(num_periods, last_known, log_callback)
            if not web_data:
                log_callback("未解析到有效的 3 位开奖数据，请检查接口返回")
                return

            existing = set(df['Period'])
            new_entries = [e for e in web_data if e['Period'] not in existing]
            if new_entries:
                updated_df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
                updated_df = updated_df.drop_duplicates(subset=['Period'], keep='first')
                updated_df = updated_df.sort_values(by='Period', ascending=False).reset_index(drop=True)
                # 原子写回：先写临时文件再 os.replace。直接覆盖原文件曾被
                # 杀软/索引器独占导致 PermissionError，且失败可能损坏原文件。
                # 临时文件保留 .xlsx 后缀（pandas 按扩展名校验写入引擎）。
                tmp_path = f'{self.file_path}.{uuid.uuid4().hex[:8]}.tmp.xlsx'
                try:
                    updated_df.to_excel(tmp_path, index=False, engine='openpyxl')
                    os.replace(tmp_path, self.file_path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                log_callback(f"已更新 {len(new_entries)} 条记录，最新期号 "
                             f"{max(e['Period'] for e in new_entries)}")
                merged_df = updated_df
            else:
                log_callback("没有新数据需要更新")
                merged_df = df
            # 用合并后的 DataFrame 刷新内存，避免再读一次 Excel；
            # 写回已成功，若解析失败原文件完好，只需重试。
            try:
                self._apply_dataframe(merged_df, self._last_num_rows, log_callback)
            except Exception as e:
                log_callback(f"数据文件已写入但内存加载失败（{e}），请重试「更新数据」")
        except Exception as e:
            log_callback(f"更新失败：{str(e)}")

    def save_state(self, save_dir, log_callback):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'match_scaler': self.match_scaler,
                         'feature_scaler': self.feature_scaler}, f)
        with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
            pickle.dump({'data': self.data, 'features': self.features,
                         'last_period': self.last_period,
                         'recent_samples': self.recent_samples}, f)
        log_callback("数据状态已保存")

    def load_state(self, save_dir, log_callback):
        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        data_path = os.path.join(save_dir, 'data.pkl')
        if not (os.path.exists(scaler_path) and os.path.exists(data_path)):
            return False
        try:
            with open(scaler_path, 'rb') as f:
                s = pickle.load(f)
            self.scaler = s['scaler']
            self.match_scaler = s['match_scaler']
            self.feature_scaler = s['feature_scaler']
            with open(data_path, 'rb') as f:
                d = pickle.load(f)
            self.data = d['data']
            self.features = d['features']
            self.last_period = d['last_period']
            # 旧版 pkl 里的 group_data / period_to_index 已废弃，读入时忽略。
            self.recent_samples = d.get('recent_samples')
            self._matrix_cache = None
            log_callback("已加载数据与归一化状态")
            return True
        except Exception as e:
            log_callback(f"加载数据状态失败：{str(e)}")
            return False
