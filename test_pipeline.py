"""LotteryPredictor 端到端快速测试（不依赖交互，训练用缩短的 2 epochs）。

用法：
    C:/PythonProject/.venv/Scripts/python.exe test_pipeline.py

覆盖：数据加载与坏数据校验、lag 匹配数方向、滑窗标签对齐、模型构建、
快速训练与预测、状态保存/加载、真实 GUI 后台线程流程、体彩接口解析
（离线 mock 的分页/截取/坏行/早停/降级）、真实增量更新写回、全量数据
完整性、从任意目录启动真实入口的冒烟测试、回测框架的无未来泄漏与统计
口径、GUI 命中率回测按钮流程、多注选号（直选/组选6/组选3）的不重复性
与理论账目对照、命中判定函数、命中模拟器（抽样空间/账目/自检锚点）、
投注对账（跨年跳变/无新开奖/文件往返）、GUI 预算规划与命中模拟流程、
模型变体（sumspan）对照。
"""
import json
import os
import shutil
import sys
import tempfile
import time
import tkinter as tk
import warnings
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(('PASS' if cond else 'FAIL'), '|', name, ('| ' + str(detail) if detail else ''))


def main():
    import main as M
    from data_processor import DataProcessor, raw_match_counts
    from lottery_model import LotteryModel

    excel = M._resolve('历史数据.xlsx')
    logs = []
    log = lambda m: logs.append(str(m))

    # ---- 1) 数据层 ----
    dp = DataProcessor(excel, '')
    ok = dp.load_data(200, log)
    top200 = pd.read_excel(excel, engine='openpyxl')['Period'].astype(int).sort_values().tail(200)
    check('数据加载(200期)', ok and dp.data.shape == (200, 3),
          f'{top200.iloc[0]} -> {dp.last_period}')
    check('时间升序(旧→新)', ok and top200.iloc[0] < top200.iloc[-1])
    rs = dp.recent_samples
    check('recent_samples:增强列齐全(Group/OddEven/BigSmall/Sum)',
          all(c in rs.columns for c in ('Group', 'OddEven', 'BigSmall', 'Sum'))
          and len(rs) == 50 and all(':' in v for v in rs['OddEven']))
    check('数据层瘦身(group_data/period_to_index已移除)',
          not hasattr(dp, 'group_data') and not hasattr(dp, 'period_to_index'))

    import openpyxl
    bad_dir = tempfile.mkdtemp()
    bad = Path(bad_dir) / 'bad.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Period', 'Numbers'])
    ws.append([1, '1 2'])
    wb.save(bad)
    check('坏数据拒绝(非3位数字)', DataProcessor(str(bad), '').load_data(10, log) is False)
    shutil.rmtree(bad_dir, ignore_errors=True)

    mc = raw_match_counts(np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3]], dtype=np.int32))
    check('lag匹配数方向(与更旧期比较)',
          np.array_equal(mc, [[0, 0, 0], [3, 0, 0], [0, 0, 0], [0, 3, 3]]),
          str(mc.astype(int).tolist()))

    mat = dp.build_feature_matrix(True)
    check('特征矩阵(200,9,float32)', mat.shape == (200, 9) and mat.dtype == np.float32)
    mat_upto = dp.build_feature_matrix(False, upto=87)
    check('特征矩阵upto等价(前缀切片)',
          mat_upto.shape == (87, 9) and np.allclose(mat_upto, mat[:87]),
          f'{mat_upto.shape} vs {mat[:87].shape}')
    check('特征矩阵upto边界(0/超长)',
          dp.build_feature_matrix(False, upto=0).shape == (0, 9)
          and dp.build_feature_matrix(False, upto=99999).shape == (200, 9))

    # ---- 1b) 回测框架：无未来泄漏 / 统计口径 ----
    import evaluation as EV

    rng = np.random.default_rng(7)
    synth = rng.integers(0, 10, (120, 3)).astype(np.int32)

    class Spy(EV.Strategy):
        name = 'spy'

        def __init__(self):
            self.seen = []
            self.content_ok = True

        def predict(self, history):
            self.seen.append(len(history))
            if not np.array_equal(history, synth[:len(history)]):
                self.content_ok = False
            return np.array([0, 0, 0])

    spy = Spy()
    EV.run_backtest(synth, [spy], start=60)
    check('回测无未来泄漏(逐期只见data[:t])',
          spy.seen == list(range(60, 120)) and spy.content_ok,
          f'{len(spy.seen)}期')

    class Oracle(EV.Strategy):
        name = 'oracle'

        def predict(self, history):
            return synth[len(history)].copy()

    oracle = EV.run_backtest(synth, [Oracle()], start=60)['oracle']
    check('回测统计口径(全知策略100%命中)',
          oracle['digit_rate'] == 1.0 and oracle['exact_hits'] == 60
          and oracle['set_hits'] == 60)

    lo, hi = EV.wilson_interval(10, 100)
    rand_stat = EV.run_backtest(synth, [EV.RandomStrategy(3)], start=60)
    check('Wilson区间/二项检验/随机基线',
          lo < 0.1 < hi
          and abs(EV.binomtest(10, 100, 0.1).pvalue - 1.0) < 1e-6
          and abs(rand_stat['随机选号(基线)']['digit_rate'] - 0.1) < 0.06)

    hot = EV.HotStrategy()
    cold = EV.ColdStrategy()
    EV.run_backtest(synth, [hot, cold], start=60)
    ref_counts = np.stack([np.bincount(synth[:, i], minlength=10) for i in range(3)])
    check('热/冷号:增量计数==全量bincount',
          np.array_equal(hot.counts, ref_counts)
          and np.array_equal(cold.counts, ref_counts)
          and np.array_equal(hot.counts.argmax(axis=1),
                             ref_counts.argmax(axis=1))
          and np.array_equal(cold.counts.argmin(axis=1),
                             ref_counts.argmin(axis=1)))

    lm = LotteryModel(50, dp)
    X, y = lm.preprocess_data(log, fit_scalers=True)
    check('滑窗标签对齐 y[i]==data[i+50]',
          X.shape == (150, 50, 9) and y.shape == (150, 3)
          and np.array_equal(y, dp.data[50:200]))
    check('滑窗零拷贝等价(窗口==矩阵切片)',
          np.allclose(X[0], mat[:50]) and np.allclose(X[7], mat[7:57]))

    # ---- 2) 模型层 ----
    import keras_tuner as kt
    m1 = lm.build_model()
    m2 = lm.build_model(kt.HyperParameters())
    check('模型构建(默认+hp分支)', m1.output_shape == (None, 3, 10) and m2.output_shape == (None, 3, 10))

    split = int(len(X) * 0.8)
    h = m1.fit(X[:split], y[:split], epochs=2, batch_size=16,
               validation_data=(X[split:], y[split:]), verbose=0)
    loss = float(h.history['loss'][-1])
    check('快速训练(2 epochs)', np.isfinite(loss) and loss < 10, f'loss={loss:.3f}')

    lm.model = m1
    pred = lm.predict_next(log)
    check('预测为3位数字', isinstance(pred, str) and len(pred) == 3 and pred.isdigit(), pred)
    check('训练/预测特征矩阵一致', np.allclose(mat, dp.build_feature_matrix(False)))

    # ---- 2b) 多注选号：不重复是 k/1000 覆盖率成立的前提 ----
    import evaluation
    picks5 = lm.predict_multiple(5, log)
    check('多注选号(5注不重复且格式正确)',
          picks5 is not None and len(picks5) == 5
          and len({p['numbers'] for p in picks5}) == 5
          and all(len(p['numbers']) == 3 and p['numbers'].isdigit() for p in picks5),
          str([p['numbers'] for p in picks5]) if picks5 else 'None')
    check('多注选号第1注=predict_next',
          picks5 is not None and picks5[0]['numbers'] == pred,
          f"{picks5[0]['numbers'] if picks5 else '-'} vs {pred}")
    check('多注选号按联合概率降序',
          picks5 is not None and all(picks5[i]['score'] >= picks5[i + 1]['score']
                                     for i in range(4)))
    check('多注选号拒绝非法注数',
          lm.predict_multiple(0, log) is None and lm.predict_multiple(1001, log) is None
          and lm.predict_multiple('abc', log) is None)
    odds5 = evaluation.multi_bet_odds(5)
    check('多注账目(5注=0.5%覆盖/10元成本)',
          abs(odds5['coverage'] - 0.005) < 1e-12
          and abs(odds5['cost'] - 10.0) < 1e-12
          and abs(odds5['expected_return'] - 5.2) < 1e-12,
          f"覆盖率{odds5['coverage_pct']:.2f}% 返奖率{odds5['return_rate'] * 100:.0f}%")
    o_pay = evaluation.multi_bet_odds(5, payout_per_hit=2000)
    check('多注账目:自定义奖金覆盖分支',
          abs(o_pay['expected_return'] - 10.0) < 1e-9
          and abs(o_pay['return_rate'] - 1.0) < 1e-9
          and o_pay['payout'] == 1040.0)

    # ---- 2c) 组选玩法：命中频率更高，但期望回报率与直选相同 ----
    picks6 = lm.predict_multiple(5, log, '组选6')
    check('组选6选号(5注、三位互不相同、不重复、降序)',
          picks6 is not None and len(picks6) == 5
          and len({p['numbers'] for p in picks6}) == 5
          and all(len(set(p['numbers'])) == 3
                  and p['numbers'] == ''.join(sorted(p['numbers']))
                  for p in picks6),
          str([p['numbers'] for p in picks6]) if picks6 else 'None')
    picks3 = lm.predict_multiple(5, log, '组选3')
    check('组选3选号(5注、恰一对相同数字)',
          picks3 is not None and len(picks3) == 5
          and all(len(set(p['numbers'])) == 2 for p in picks3),
          str([p['numbers'] for p in picks3]) if picks3 else 'None')
    check('组选玩法参数与上限校验',
          lm.predict_multiple(5, log, '排列5') is None
          and lm.predict_multiple(121, log, '组选6') is None
          and lm.predict_multiple(91, log, '组选3') is None)
    o6, o3 = evaluation.play_odds('组选6', 5), evaluation.play_odds('组选3', 5)
    check('组选账目(组选6命中3%/组选3为1.5%/返奖率一致)',
          abs(o6['coverage'] - 0.03) < 1e-12
          and abs(o3['coverage'] - 0.015) < 1e-12
          and abs(o6['return_rate'] - 0.519) < 1e-9
          and abs(o3['expected_return'] - 5.19) < 1e-9,
          f"组选6命中{o6['coverage_pct']:.1f}% 组选3命中{o3['coverage_pct']:.1f}%")
    plan = evaluation.budget_plan(100)
    check('预算计划(100元=50注三玩法对照)',
          plan['bets'] == 50
          and abs(plan['modes']['组选6']['coverage'] - 0.3) < 1e-12
          and abs(plan['modes']['直选']['return_rate'] - 0.52) < 1e-12)

    # ---- 2d) 命中模拟与投注对账 ----
    import settlement as ST
    import simulation as SIM
    check('命中判定:直选顺序敏感',
          evaluation.bet_hits('123', '123', '直选') and not evaluation.bet_hits('123', '321', '直选'))
    check('命中判定:组选6规范化',
          evaluation.bet_hits('149', '941', '组选6') and not evaluation.bet_hits('114', '941', '组选6')
          and not evaluation.bet_hits('149', '941', '组选3'))
    check('命中判定:组选3恰一对',
          evaluation.bet_hits('449', '944', '组选3') and not evaluation.bet_hits('459', '944', '组选3')
          and not evaluation.bet_hits('444', '449', '组选3'))
    try:
        evaluation.bet_hits('123', '123', '排列5')
        ok_mode = False
    except ValueError:
        ok_mode = True
    check('命中判定:非法mode抛错', ok_mode)

    rng_sim = np.random.default_rng(0)
    space6 = SIM.sample_bets(rng_sim, '组选6', 120)
    check('随机抽样:空间对齐且不重复',
          len(space6) == 120 and len(set(space6)) == 120
          and all(len(set(x)) == 3 and x == ''.join(sorted(x)) for x in space6)
          and len(SIM.sample_bets(rng_sim, '组选3', 90)) == 90)

    # 组合枚举三处共用 iter_combos：口径必须一致
    from lottery_model import iter_combos
    combos6_ref = [f'{a}{b}{c}' for a in range(10) for b in range(a, 10)
                   for c in range(b, 10) if len({a, b, c}) == 3]
    check('iter_combos:三处枚举口径一致',
          list(iter_combos('组选6')) == combos6_ref
          and len(list(iter_combos('组选3'))) == 90
          and len(list(iter_combos('直选'))) == 1000
          and SIM._all_bets('组选6') == combos6_ref
          and SIM.sample_bets(np.random.default_rng(3), '组选6', 120) == combos6_ref
          and SIM.sample_bets(np.random.default_rng(4), '直选', 1000) == list(iter_combos('直选')))
    try:
        list(iter_combos('排列5'))
        bad_iter_mode = False
    except ValueError:
        bad_iter_mode = True
    check('iter_combos:非法mode抛错', bad_iter_mode)

    from lottery_model import rank_combos
    uni = np.full((3, 10), 0.1)
    r_all6 = rank_combos(uni, '组选6', 120)
    r_d1000 = rank_combos(uni, '直选', 1000)
    check('rank_combos:上限全收(120/1000注不重复)',
          len(r_all6) == 120 and len({p['numbers'] for p in r_all6}) == 120
          and len(r_d1000) == 1000 and len({p['numbers'] for p in r_d1000}) == 1000)

    sim_rng = np.random.default_rng(1)
    sim_data = np.stack([sim_rng.integers(0, 10, 3)
                         for _ in range(300)]).astype(np.int32)
    r120 = SIM.simulate_random(sim_data, '组选6', 120, start=0, seed=0)
    check('模拟器:120注组选6命中≈0.72', abs(r120['hit_rate'] - 0.72) < 0.06,
          f"实测 {r120['hit_rate']:.3f}")
    r1000 = SIM.simulate_random(sim_data[:20], '直选', 1000, start=0, seed=0)
    check('模拟器:1000注直选每期必中账目精确',
          r1000['hit_draws'] == 20
          and abs(r1000['pnl'] - (20 * 1040 - 20 * 2000)) < 1e-9)
    check('模拟器:期望命中与统计字段',
          abs(r120['expected_hits'] - r120['n'] * 0.72) < 1e-9
          and 0 <= r120['ci'][0] <= r120['ci'][1] <= 1
          and 0 <= r120['p'] <= 1
          and SIM.longest_false_streak(r120['win_flags']) == r120['longest_lose_streak'])
    check('模拟器:报告含理论值与52%声明',
          '理论命中率' in SIM.format_simulation_report(r120)
          and '52%' in SIM.format_simulation_report(r120))
    check('最长连败:单一实现(SIM/settlement共用EV)',
          SIM.longest_false_streak is EV.longest_false_streak
          and EV.longest_false_streak([False, False, True, False]) == 2
          and EV.longest_false_streak([]) == 0)

    var_results = SIM.run_variant_backtest(sim_data, variants=('sumspan',),
                                           eval_len=30, epochs=2, log=log)
    vr = var_results['sumspan']
    check('模型变体:sumspan全流程跑通',
          vr['n'] == 30 and 0 <= vr['digit_rate'] <= 1 and 0 <= vr['exact_p'] <= 1,
          f"逐位 {vr['digit_rate']:.3f} 直选 {vr['exact_hits']}/30")
    check('模型变体:报告含随机期望与p值',
          '随机期望' in SIM.format_variant_report(var_results)
          and 'exact_p' in vr)

    settle_dir = tempfile.mkdtemp()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Period', 'Numbers'])
    for pp, nn in [(25360, '1 2 3'), (26001, '5 6 7'), (26002, '7 6 5'),
                   (26003, '9 9 4')]:
        ws.append([pp, nn])
    st_excel = Path(settle_dir) / 's.xlsx'
    wb.save(st_excel)
    p_picks = ST.save_picks(25360, '组选6',
                            [{'numbers': '567', 'score': 0.01},
                             {'numbers': '149', 'score': 0.009}],
                            dir_path=settle_dir)
    rep = ST.settle(p_picks, str(st_excel))
    check('对账:跨年跳变不依赖base+1',
          rep['draws_checked'] == 3 and rep['hit_draws'] == 2
          and rep['return_total'] == 346 and abs(rep['cost_total'] - 12.0) < 1e-9
          and rep['longest_lose_streak'] == 1,
          f"命中 {rep['hit_draws']}/3")
    p2 = ST.save_picks(26003, '组选6', [{'numbers': '567', 'score': 0.01}],
                       dir_path=settle_dir)
    rep0 = ST.settle(p2, str(st_excel))
    check('对账:无新开奖友好提示',
          rep0['draws_checked'] == 0 and '暂无新开奖' in ST.format_settlement(rep0))
    rec_round = ST.load_picks(p_picks)
    check('对账:picks文件往返',
          rec_round['picks'] == ['567', '149'] and rec_round['mode'] == '组选6'
          and 'theoretical' in rec_round)
    rep_df = ST.settle(p_picks, str(st_excel),
                       df=pd.read_excel(st_excel, engine='openpyxl'))
    check('对账:预读df复用结果一致',
          rep_df['hit_draws'] == rep['hit_draws'] and rep_df['pnl'] == rep['pnl']
          and rep_df['longest_lose_streak'] == rep['longest_lose_streak'])
    shutil.rmtree(settle_dir, ignore_errors=True)

    # ---- 2e) 模型选号器 / load_state 形状校验 / 训练中断回调 ----
    import tensorflow as tf
    import lottery_model as LMK
    tiny = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 9)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(30),
        tf.keras.layers.Reshape((3, 10)),
        tf.keras.layers.Softmax(axis=-1),
    ])
    tiny.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())
    orig_build = LMK.LotteryModel.build_model
    LMK.LotteryModel.build_model = lambda self, hp=None: tiny
    try:
        dp_mp = DataProcessor(excel, '')
        dp_mp.load_data(200, log)
        res_m = SIM.simulate_model(dp_mp, '组选6', 5, eval_len=10, epochs=1,
                                   seq_len=50, log=log)
        check('模型选号器:simulate_model全流程',
              res_m['n'] == 10 and 'hit_rate' in res_m and 'ci' in res_m
              and abs(res_m['theoretical_rate'] - 0.03) < 1e-12
              and res_m['payout'] == 173 and res_m['bets'] == 5,
              f"命中 {res_m['hit_draws']}/{res_m['n']}")
        picker = SIM.ModelPicker(dp_mp, 180, seq_len=50, epochs=1, log=log)
        picks_pk = picker.pick(190, '直选', 5)
        check('模型选号器:缓存矩阵与逐期重算等价',
              np.allclose(picker.matrix[:190], dp_mp.build_feature_matrix(False, upto=190))
              and len({p['numbers'] for p in picks_pk}) == 5
              and all(len(p['numbers']) == 3 and p['numbers'].isdigit() for p in picks_pk))
    finally:
        LMK.LotteryModel.build_model = orig_build

    shape_dir = tempfile.mkdtemp()
    bad_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)),
                                     tf.keras.layers.Dense(10, activation='softmax')])
    bad_model.save(str(Path(shape_dir) / 'model_bad.h5'))
    check('模型加载:输出形状不匹配拒绝加载',
          not LotteryModel(50, dp).load_state(shape_dir, log)
          and any('输出结构' in m for m in logs))
    shutil.rmtree(shape_dir, ignore_errors=True)

    probe = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    cb = LMK._InterruptCallback(lambda: True)
    cb.set_model(probe)
    cb.on_epoch_end(0)
    check('训练中断:epoch级置停止标志', cb.triggered and bool(probe.stop_training))
    cb2 = LMK._InterruptCallback(lambda: True, raise_on_interrupt=True)
    raised_int = False
    try:
        cb2.on_epoch_end(0)
    except LMK._InterruptedError:
        raised_int = True
    check('训练中断:search中抛Fatal终止搜索', raised_int)

    # ---- 3) 状态往返 ----
    state_dir = tempfile.mkdtemp()
    dp.save_state(state_dir, log)
    lm.save_state(state_dir, log)
    dp3 = DataProcessor(excel, '')
    lm3 = LotteryModel(50, dp3)
    a = dp3.load_state(state_dir, log)
    b = lm3.load_state(state_dir, log)
    check('状态往返后预测一致', a and b and lm3.predict_next(log) == pred)
    check('序列长度不匹配时拒绝加载', not LotteryModel(10, dp3).load_state(state_dir, log))
    shutil.rmtree(state_dir, ignore_errors=True)

    # ---- 4) 真实 GUI 后台流程（Hyperband 替换为快速训练）----
    import gui as gui_mod
    from gui import LotteryPredictorGUI
    gui_mod.messagebox.showinfo = lambda *a, **k: None
    gui_mod.messagebox.showwarning = lambda *a, **k: print('  [GUI提示]', a[1] if len(a) > 1 else '')
    errors = []
    gui_mod.messagebox.showerror = lambda *a, **k: errors.append(a)

    def fast_train(self, log_callback, interrupt_flag, progress_callback=None):
        Xf, yf = self.preprocess_data(log_callback, fit_scalers=True)
        if Xf is None:
            return False
        sp = int(len(Xf) * 0.8)
        self.model = self.build_model()
        self.model.fit(Xf[:sp], yf[:sp], epochs=2, batch_size=16,
                       validation_data=(Xf[sp:], yf[sp:]), verbose=0)
        return True

    LotteryModel.train = fast_train

    gdir = tempfile.mkdtemp()
    root = tk.Tk()
    root.withdraw()
    dp_gui = DataProcessor(excel, '')
    g = LotteryPredictorGUI(root, dp_gui, LotteryModel(50, dp_gui), gdir)
    g.num_rows.set('200')

    t0 = time.time()
    g.initial_run()
    submit_ms = (time.time() - t0) * 1000
    deadline = time.time() + 180
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    text = g.result_text.get('1.0', 'end')
    check('GUI初次运行(非阻塞+完成+不给预测号码)',
          not g.is_running and not errors and '不给出「下一期号码」' in text
          and text.count('预测下一期') == 0,
          f'按钮回调耗时 {submit_ms:.1f}ms')
    check('GUI运行后状态已保存',
          any(Path(gdir).glob('model_*.h5')) and (Path(gdir) / 'scalers.pkl').exists())

    g.continue_run()
    deadline = time.time() + 60
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    text = g.result_text.get('1.0', 'end')
    check('GUI继续运行(加载状态出结论,仍无预测号码)',
          not g.is_running and not errors and text.count('运行完成') == 2
          and text.count('预测下一期') == 0)

    g.pick_count.set('5')
    g.multi_pick_run()
    deadline = time.time() + 60
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    mp_text = g.result_text.get('1.0', 'end')
    check('GUI直选选号(5注+覆盖率+成本说明)',
          not g.is_running and not errors and '5 注直选' in mp_text
          and '0.50%' in mp_text and '期望回报' in mp_text,
          f"结果区含账目: {'期望回报' in mp_text}")
    check('GUI直选选号(打分不冒充中奖概率)',
          mp_text.count('模型打分') >= 5 and '每注真实中奖概率恒为' in mp_text
          and '   中奖概率' not in mp_text,
          f"模型打分标签 {mp_text.count('模型打分')} 处")

    g.pick_mode.set('组选6')
    g.multi_pick_run()
    deadline = time.time() + 60
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    mp2 = g.result_text.get('1.0', 'end')
    check('GUI组选6选号(5注+命中3%+奖金173)',
          not g.is_running and not errors and '5 注组选6' in mp2
          and '3.00%' in mp2 and '173' in mp2)
    g.pick_mode.set('直选')

    import settlement as ST
    ST.PICKS_DIR = Path(tempfile.mkdtemp())   # 存档指向临时目录，不污染项目
    g.pick_count.set('5')
    g.multi_pick_run()
    deadline = time.time() + 60
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    picks_files = list(ST.PICKS_DIR.glob('picks_*.json'))
    picks_ok = False
    if picks_files:
        rec = json.loads(picks_files[-1].read_text(encoding='utf-8'))
        picks_ok = rec['base_period'] == dp_gui.last_period and rec['mode'] == '直选'
    check('GUI:生成选号自动存档picks',
          not g.is_running and not errors and picks_ok,
          f"{len(picks_files)} 份存档")

    g.budget_run()   # budget 默认 '100' → 50 注直选
    deadline = time.time() + 90
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    bp_text = g.result_text.get('1.0', 'end')
    check('GUI:预算规划(对照表+选号)',
          not g.is_running and not errors and '预算规划' in bp_text
          and '返奖率' in bp_text and '100 元 = 50 注' in bp_text,
          f"含对照: {'组选6' in bp_text}")

    g.simulate_run()   # sim_bets 默认 30、sim_mode 默认组选6
    deadline = time.time() + 300
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    sm_text = g.result_text.get('1.0', 'end')
    log_text = g.progress_text.get('1.0', 'end')
    check('GUI:命中模拟(回放+理论对照)',
          not g.is_running and not errors and '【命中模拟】' in sm_text
          and '理论期望' in sm_text and '实际命中' in log_text
          and '52%' in log_text)

    t0 = time.time()
    g.backtest_run()
    bt_ms = (time.time() - t0) * 1000
    deadline = time.time() + 120
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    bt_text = g.result_text.get('1.0', 'end')
    check('GUI命中率回测(非阻塞+基线结论)',
          not g.is_running and '回测结论' in bt_text
          and '显著优于随机' in bt_text,
          f'按钮回调耗时 {bt_ms:.1f}ms')

    # 新增的严谨性检验按钮
    g.randomness_run()
    deadline = time.time() + 120
    while g.is_running and time.time() < deadline:
        root.update()
        time.sleep(0.03)
    rnd_text = g.result_text.get('1.0', 'end')
    check('GUI随机性检验(非阻塞+输出结论)',
          not g.is_running and '随机性检验' in rnd_text and not errors)
    check('GUI累积审计已接线',
          hasattr(g, 'audit_run') and hasattr(g, '_do_audit'))

    # GUI 更新数据链路：mock 接口写入临时副本，不动真实 Excel
    from unittest.mock import patch as _patch
    fetch_cfg = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
    fetch_xlsx = Path(gdir) / 'fetch_copy.xlsx'
    shutil.copy(excel, fetch_xlsx)
    real_dp_gui = g.data_processor
    g.data_processor = DataProcessor(str(fetch_xlsx), fetch_cfg['fetch_url'])

    def gui_fake_page(self, page_no, log_callback=None):
        return [{'lotteryDrawNum': str(p),
                 'lotteryDrawResult': f'{p % 10} {(p + 1) % 10} {(p + 2) % 10} 7 8'}
                for p in range(400, 300, -1)]

    g.fetch_periods.set('5')
    with _patch.object(DataProcessor, '_fetch_api_page', gui_fake_page):
        g.fetch_and_update()
        deadline = time.time() + 60
        while g.is_running and time.time() < deadline:
            root.update()
            time.sleep(0.03)
        first_fetch_ok = (not g.is_running and not errors
                          and g.data_processor.last_period == real_dp_gui.last_period)
        g.fetch_and_update()   # 二次更新：全部已存在 → 仅提示
        deadline = time.time() + 60
        while g.is_running and time.time() < deadline:
            root.update()
            time.sleep(0.03)
    fetch_log_text = g.progress_text.get('1.0', 'end')
    check('GUI更新数据链路(mock抓取+二次无新数据)',
          first_fetch_ok and not g.is_running and not errors
          and '已更新' in fetch_log_text
          and '没有新数据需要更新' in fetch_log_text,
          f"first={first_fetch_ok} 已更新={'已更新' in fetch_log_text} "
          f"无新数据={'没有新数据需要更新' in fetch_log_text} "
          f"错误数={len(errors)}")
    g.data_processor = real_dp_gui

    g.interrupt_prediction()
    check('GUI中断按钮(标志位生效)', g.interrupt is True)
    g.interrupt = False
    root.destroy()
    shutil.rmtree(gdir, ignore_errors=True)

    # ---- 5) 体彩数据源：离线 mock 解析 + 真实增量更新 ----
    import subprocess
    from unittest.mock import patch

    config = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
    calls = []

    def fake_page(self, page_no, log_callback=None):
        """模拟接口分页：每页 100 条 5 位排列5 号码，前三位应为排列3。"""
        calls.append(page_no)
        starts = {1: 400, 2: 300, 3: 200}
        if page_no not in starts:
            return []
        start = starts[page_no]
        rows = [{'lotteryDrawNum': str(p),
                 'lotteryDrawResult': f'{p % 10} {(p + 1) % 10} {(p + 2) % 10} 8 9'}
                for p in range(start, start - 100, -1)]
        if page_no == 1:
            rows += [
                {'lotteryDrawNum': '999', 'lotteryDrawResult': '1 2'},          # 位数不足
                {'lotteryDrawNum': '998', 'lotteryDrawResult': 'x y z 8 9'},    # 非数字
                {'lotteryDrawNum': 'abc', 'lotteryDrawResult': '1 2 3 8 9'},    # 期号非法
                {'lotteryDrawNum': '997'},                                       # 缺号码字段
            ]
        return rows

    dp_mock = DataProcessor(excel, config['fetch_url'])
    with patch.object(DataProcessor, '_fetch_api_page', fake_page):
        recs = dp_mock._fetch_api_data(500, 150, lambda m: None)
        pages_first = list(calls)
        calls.clear()
        early = dp_mock._fetch_api_data(500, 350, lambda m: None)
        pages_early = list(calls)
    nums_ok = all(len(r['Numbers'].split()) == 3
                  and all(t.isdigit() and len(t) == 1 for t in r['Numbers'].split())
                  for r in recs)
    bad_present = {999, 998, 997} & {r['Period'] for r in recs}
    check('接口解析:排列5前3位截取+分页+坏行跳过',
          len(recs) == 300 and nums_ok and not bad_present
          and recs[0]['Period'] == 400 and recs[-1]['Period'] == 101
          and len({r['Period'] for r in recs}) == 300
          and pages_first == [1, 2, 3],
          f'{len(recs)}条 / 分页{pages_first}')
    check('接口解析:覆盖本地最新期即早停', len(early) == 100 and pages_early == [1],
          f'{len(early)}条 / 分页{pages_early}')

    # 接口返回空 / 异常时不得破坏原文件
    tmp_dir = tempfile.mkdtemp()
    tmp_xlsx = Path(tmp_dir) / '历史数据.xlsx'
    shutil.copy(excel, tmp_xlsx)
    before_bytes = tmp_xlsx.read_bytes()
    dp_empty = DataProcessor(str(tmp_xlsx), config['fetch_url'])
    with patch.object(DataProcessor, '_fetch_api_page', lambda *a, **k: []):
        dp_empty.fetch_and_update(5, log)
    noop_ok = '未解析到有效的' in ' '.join(logs[-4:])
    fail_logs = []
    with patch.object(DataProcessor, '_fetch_api_page',
                      side_effect=RuntimeError('mock 网络故障')):
        dp_empty.fetch_and_update(5, fail_logs.append)
    check('接口空响应/异常降级且原文件不变',
          noop_ok and any('更新失败' in m for m in fail_logs)
          and tmp_xlsx.read_bytes() == before_bytes,
          f'空响应={noop_ok} 异常已捕获={any("更新失败" in m for m in fail_logs)}')

    # 写回原子化：to_excel 失败时原文件必须完好且无残留临时文件
    atomic_logs = []
    before_atomic = tmp_xlsx.read_bytes()

    def failing_to_excel(self, path, *a, **k):
        raise PermissionError('mock 写入被独占')

    with patch.object(DataProcessor, '_fetch_api_page', fake_page), \
         patch.object(pd.DataFrame, 'to_excel', failing_to_excel):
        dp_atomic = DataProcessor(str(tmp_xlsx), config['fetch_url'])
        dp_atomic.fetch_and_update(5, atomic_logs.append)
    leftovers = [p.name for p in Path(tmp_dir).glob('*.tmp*')]
    check('Excel原子写回(写失败原文件完好无残留)',
          tmp_xlsx.read_bytes() == before_atomic
          and any('更新失败' in m for m in atomic_logs) and not leftovers,
          f'残留={leftovers}')

    # paths.py：路径解析与配置读取一致（CLI 不再为路径导入 TF）
    import paths
    check('paths模块:resolve与config读取一致',
          paths.resolve('历史数据.xlsx') == M._resolve('历史数据.xlsx')
          and paths.load_config()['fetch_url'] == config['fetch_url'])

    # settlement CLI 冒烟：默认路径走 paths.py（此前 import main 会连带 TF）
    cli_dir = tempfile.mkdtemp()
    try:
        cli_picks = ST.save_picks(30000, '直选', [{'numbers': '123', 'score': 0.01}],
                                  dir_path=cli_dir)
        env_c = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='3', PYTHONIOENCODING='utf-8')
        proc_c = subprocess.run([sys.executable, str(BASE / 'settlement.py'),
                                 '--file', str(cli_picks)],
                                capture_output=True, timeout=120, env=env_c,
                                cwd=str(BASE.parent))
        out_c = proc_c.stdout.decode('utf-8', 'replace')
        check('settlement CLI冒烟(默认路径走paths)',
              proc_c.returncode == 0 and '暂无新开奖' in out_c,
              out_c.strip()[:60] or proc_c.stderr.decode('utf-8', 'replace')[:80])
    finally:
        shutil.rmtree(cli_dir, ignore_errors=True)

    # 真实接口：把副本的最新 3 期删掉，验证能增量写回且号码与权威数据一致
    full_df = pd.read_excel(excel, engine='openpyxl')
    latest3 = full_df.head(3).copy()
    full_df.iloc[3:].to_excel(tmp_xlsx, index=False, engine='openpyxl')
    dp_inc = DataProcessor(str(tmp_xlsx), config['fetch_url'])
    inc_logs = []
    dp_inc.fetch_and_update(20, inc_logs.append)
    after = pd.read_excel(tmp_xlsx, engine='openpyxl')
    merged = dict(zip(after['Period'], after['Numbers']))
    restored = all(merged.get(int(p)) == n for p, n in
                   zip(latest3['Period'], latest3['Numbers']))
    check('真实接口增量更新写回(缺3期→补齐)',
          any('已更新' in m for m in inc_logs) and len(after) >= len(full_df)
          and restored and int(after['Period'].max()) == dp_inc.last_period,
          ' '.join(inc_logs[:2]))

    # 接口返回的期号全部已存在 → 不写文件、内存仍刷新
    dup_logs = []
    before_dup = tmp_xlsx.read_bytes()

    def dup_page(self, page_no, log_callback=None):
        return [{'lotteryDrawNum': str(int(after['Period'].max())),
                 'lotteryDrawResult': '1 2 3 4 5'}]

    with patch.object(DataProcessor, '_fetch_api_page', dup_page):
        dp_dup = DataProcessor(str(tmp_xlsx), config['fetch_url'])
        dp_dup.fetch_and_update(5, dup_logs.append)
    check('接口数据全部已存在(不写文件仅提示)',
          any('没有新数据需要更新' in m for m in dup_logs)
          and tmp_xlsx.read_bytes() == before_dup)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- 6) 更新后全量数据完整性（不硬编码期数，容忍真实新开奖入库）----
    periods_all = full_df['Period'].astype(int).tolist()
    nums_all = full_df['Numbers'].astype(str).tolist()
    fmt_ok = all(len(n.split()) == 3 and all(t.isdigit() and len(t) == 1
                                              for t in n.split()) for n in nums_all)
    n_all = len(periods_all)
    check('全量数据完整性(唯一/格式/降序)',
          n_all > 0 and len(set(periods_all)) == n_all and fmt_ok
          and all(periods_all[i] > periods_all[i + 1] for i in range(n_all - 1)),
          f'{n_all}期 {periods_all[0]}→{periods_all[-1]}')

    # ---- 7) 真实入口冒烟：从项目根目录启动 main.py（验证路径修复 + Tk 初始化）----
    try:
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='3')
        proc = subprocess.Popen([sys.executable, str(BASE / 'main.py')],
                                cwd=str(BASE.parent), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env)
        try:
            time.sleep(6)
            alive = proc.poll() is None
            if alive:
                proc.terminate()
            out, err = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=10)
        err_text = err.decode('utf-8', 'replace')
        check('真实入口启动(任意CWD/无异常)', alive and 'Traceback' not in err_text,
              '进程存活=是' if alive else f'提前退出 code={proc.returncode}')
    except Exception as e:
        print('WARN | 入口冒烟跳过:', e)

    # ---- 随机性检验 + Holm 多重比较校正 ----
    import randomness
    rng = np.random.default_rng(20261107)
    fake_random = rng.integers(0, 10, size=(4000, 3))
    res_r = randomness.analyze(fake_random)
    check('随机性检验:纯随机数据整体相容(Holm)', res_r['all_pass'])

    # Holm step-down 的确定性回归：排序后逐项乘 (m-rank)，取前缀最大值并截断到 1
    adj = randomness.holm_adjust([0.01, 0.04, 0.03, 0.9, 0.8])
    check('Holm校正:step-down结果精确匹配',
          np.allclose(adj, [0.05, 0.12, 0.12, 1.0, 1.0]),
          [round(float(x), 4) for x in adj])
    check('Holm校正:校正后p不小于原始p',
          bool(np.all(adj >= np.array([0.01, 0.04, 0.03, 0.9, 0.8]))))
    check('Holm校正:结果截断到1', bool(np.all(adj <= 1.0)))
    check('Holm校正:空输入不报错', len(randomness.holm_adjust([])) == 0)

    # 裸判误报应被校正吸收：纯随机数据不应再被判成「有结构」
    false_positives = []
    for seed in (1, 7, 42, 20261107):
        r = randomness.analyze(np.random.default_rng(seed).integers(
            0, 10, size=(4000, 3)))
        false_positives.append((seed, r['n_nominal'], r['n_holm'], r['all_pass']))
    check('Holm校正:纯随机数据无一被误判为有结构',
          all(n_holm == 0 and ok for _, _, n_holm, ok in false_positives)
          and any(n_raw > 0 for _, n_raw, _, _ in false_positives),
          str(false_positives))
    check('Holm校正:族错误率符合理论值',
          abs(res_r['familywise_error_raw'] - (1 - 0.95 ** 9)) < 1e-12,
          round(res_r['familywise_error_raw'], 4))
    check('Holm校正:结果含逐项p_holm与pass_holm',
          all('p_holm' in t and 'pass_holm' in t and t['p_holm'] >= t['p']
              for t in res_r['tests']))

    fake_biased = rng.integers(0, 10, size=(4000, 3))
    fake_biased[:, 0] = rng.choice(10, size=4000, p=[0.3] + [0.7 / 9] * 9)
    res_b = randomness.analyze(fake_biased)
    check('随机性检验:偏斜数据被检出(原始p)',
          any((not t['pass']) and '第1位数字均匀性' in t['name'] for t in res_b['tests']))
    check('Holm校正:真实偏斜校正后仍显著', res_b['n_holm'] >= 1,
          f"Holm显著 {res_b['n_holm']} 项")
    report_r = randomness.format_report(res_r)
    check('随机性检验:报告含结论声明', 'k/1000' in report_r)
    check('随机性检验:报告标注Holm校正', 'Holm' in report_r and '37%' in report_r)
    try:
        randomness.analyze(np.zeros((10, 3), dtype=int))
        small_ok = False
    except ValueError:
        small_ok = True
    check('随机性检验:样本过小拒绝', small_ok)

    # ---- 随机性检验的玩法通用化（不再写死 3 位 / 0-9）----
    check('随机性检验:默认自由度与旧版一致',
          [t['dof'] for t in res_r['tests']] == [9, 9, 9, 972, 81, 81, 81, 2, 27])
    d26 = rng.integers(0, 6, size=(4000, 2))
    r26 = randomness.analyze(d26, digits=2, base=6)
    check('随机性检验:(2,6)参数化可运行',
          r26['n_tests'] == 7 and r26['n_outcomes'] == 36
          and [t['dof'] for t in r26['tests']] == [5, 5, 25, 25, 25, 1, 10])
    check('随机性检验:(2,6)报告给出k/36',
          'k/36' in randomness.format_report(r26))
    b26 = d26.copy()
    b26[:, 0] = rng.choice(6, size=4000, p=[0.4] + [0.6 / 5] * 5)
    check('随机性检验:(2,6)偏斜被检出',
          randomness.analyze(b26, digits=2, base=6)['n_holm'] >= 1)
    bad_specs = 0
    for fn in (lambda: randomness.analyze(np.full((200, 3), 10)),
               lambda: randomness.analyze(np.zeros((200, 3), dtype=int),
                                          digits=2, base=10),
               lambda: randomness.analyze(d26, digits=9, base=6)):
        try:
            fn()
        except ValueError:
            bad_specs += 1
    check('随机性检验:拒绝越界/形状不符/空间超限', bad_specs == 3)

    # ---- 累积审计 ----
    import audit
    n10 = audit.required_samples(0.001, 0.10)
    check('审计:功效分析量级合理', 1e5 < n10 < 1e6, f'{n10:,.0f} 注')
    check('审计:提升越大所需样本越少',
          audit.required_samples(0.001, 0.5) < n10)
    check('审计:正态分位数求解', abs(audit._z(0.975) - 1.959964) < 1e-4,
          f'{audit._z(0.975):.6f}')
    fake_reports = [
        {'mode': '直选', 'draws_checked': 10, 'bets': 5, 'win_bets_total': 1,
         'cost_total': 100.0, 'return_total': 1040.0,
         'details': [{'period': p} for p in range(10)]},
        {'mode': '直选', 'draws_checked': 10, 'bets': 5, 'win_bets_total': 0,
         'cost_total': 100.0, 'return_total': 0.0,
         'details': [{'period': p} for p in range(10, 20)]},
    ]
    s = audit.summarize(fake_reports)['直选']
    check('审计:注级口径(100注/中1注/成本200)',
          s['bets'] == 100 and s['wins'] == 1 and abs(s['cost'] - 200) < 1e-9,
          f"bets={s['bets']} wins={s['wins']} cost={s['cost']}")
    check('审计:唯一期号去重', s['unique_periods'] == 20, s['unique_periods'])
    check('审计:实际返奖率计算', abs(s['actual_return_rate'] - 5.2) < 1e-9,
          round(s['actual_return_rate'], 4))
    check('审计:空输入给出提示', '暂无' in audit.format_audit({}))

    # ---- picks 存档口径声明 ----
    import settlement as ST
    tmpd = tempfile.mkdtemp()
    try:
        p = ST.save_picks(9999, '组选6', [{'numbers': '123', 'score': 0.01}],
                          dir_path=tmpd)
        rec = json.loads(Path(p).read_text(encoding='utf-8'))
        check('picks存档:含口径声明note',
              'note' in rec and '不是中奖概率' in rec['note']
              and '每注真实中奖概率恒为 0.60%' in rec['note'],
              rec['note'][:60])
        check('picks存档:note含该方案单期概率', '0.60%' in rec['note'],
              rec['note'][:46])
        check('picks存档:旧文件仍可加载', ST.load_picks(p)['mode'] == '组选6')
        check('picks存档:schema_version=2',
              rec.get('schema_version') == ST.PICKS_SCHEMA_VERSION == 2)
        legacy = dict(rec)
        legacy.pop('schema_version', None)
        lp = Path(tmpd) / 'legacy.json'
        lp.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf-8')
        check('picks存档:旧文件按v1兼容',
              ST.load_picks(lp)['schema_version'] == 1)
        future = dict(rec)
        future['schema_version'] = 99
        fp = Path(tmpd) / 'future.json'
        fp.write_text(json.dumps(future, ensure_ascii=False), encoding='utf-8')
        try:
            ST.load_picks(fp)
            rejected = False
        except ValueError:
            rejected = True
        check('picks存档:未来版本被拒绝', rejected)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    # ---- 脱敏聚合数据与一键复现 ----
    import re
    import reproduce
    from export_aggregate import summarize_payload

    agg_path = BASE / 'audit_aggregate.json'
    check('复现:聚合数据文件已随仓库发布', agg_path.exists())
    agg_raw = agg_path.read_text(encoding='utf-8') if agg_path.exists() else ''
    leaks = (re.findall(r'picks_\d+', agg_raw)
             + re.findall(r'20\d\d-\d\d-\d\d', agg_raw)
             + re.findall(r'\b\d{5}\b', agg_raw))
    check('复现:聚合数据不含期号/时间戳/文件名', bool(agg_raw) and not leaks,
          leaks[:3])
    if agg_raw:
        agg = json.loads(agg_raw)
        check('复现:聚合数据schema版本', agg.get('schema_version') == 1)
        all_rows = [row for blk in agg['modes'].values() for row in blk['per_draw']]
        check('复现:逐期记录只含四个数值字段',
              bool(all_rows) and all(
                  set(row) == {'bets', 'wins', 'cost', 'payout'} for row in all_rows),
              f'{len(all_rows)} 条')
        s6 = summarize_payload(agg)['组选6']
        check('复现:脱敏数据可重算出真实账目',
              s6['bets'] == 2860 and s6['wins'] == 17
              and abs(s6['pnl'] + 2779) < 1e-6
              and abs(s6['p_value'] - 0.548) < 0.001,
              f"{s6['bets']}注/{s6['wins']}中/p={s6['p_value']:.4f}")

    # 文档资产同样要守住：图、英文摘要、数据来源说明不能被悄悄删掉
    readme = (BASE / 'README.md').read_text(encoding='utf-8')
    check('文档:README嵌入两张结果图',
          '](./audit_result.png)' in readme
          and '](./kronos_migration/kronos_migration_result.png)' in readme)
    check('文档:两张结果图文件存在',
          (BASE / 'audit_result.png').is_file()
          and (BASE / 'kronos_migration' / 'kronos_migration_result.png').is_file())
    check('文档:README含英文摘要',
          '## English Summary' in readme and 'does not predict' in readme)
    check('文档:数据来源说明与config声明',
          (BASE / 'DATA_SOURCES.md').is_file()
          and 'data_source' in (BASE / 'config.json').read_text(encoding='utf-8'))

    rec = reproduce.Recorder()
    reproduce.check_randomness(rec)
    reproduce.check_audit(rec)
    n_pass = sum(1 for i in rec.items if i['ok'])
    check('复现:README声明值逐项可重算',
          rec.passed and len(rec.items) >= 45,
          f'{n_pass}/{len(rec.items)} 项')
    ds = reproduce.Recorder()
    reproduce.check_data_source(ds)
    check('复现:数据来源声明与xlsx一致',
          ds.passed and len(ds.items) >= 7,
          f'{sum(1 for i in ds.items if i["ok"])}/{len(ds.items)} 项')

    # 文档漂移必须能被检出（把 2860 注改成 2861 注）
    expected = reproduce.EXPECTED['audit']['组选6']['bets']
    try:
        reproduce.EXPECTED['audit']['组选6']['bets'] = 2861
        drift = reproduce.Recorder()
        reproduce.check_audit(drift)
        check('复现:文档数字漂移会被检出', not drift.passed,
              [i['name'] for i in drift.items if not i['ok']][:2])
    finally:
        reproduce.EXPECTED['audit']['组选6']['bets'] = expected
    restored = reproduce.Recorder()
    reproduce.check_audit(restored)
    check('复现:篡改后恢复不影响后续', restored.passed)

    # 一键复现不得拖入 TensorFlow / 模型代码（干净 clone 也能跑）。
    # 用 AST 检查真实 import 语句，不能只搜源码文本——文档字符串里
    # 「不需要 TensorFlow」这句话会被误判成依赖。
    import ast
    tree = ast.parse((BASE / 'reproduce.py').read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    banned = {'tensorflow', 'keras', 'lottery_model', 'simulation', 'keras_tuner'}
    check('复现:不依赖TensorFlow与模型模块',
          not (imported & banned),
          '顶层导入=' + ', '.join(sorted(imported)))

    # ---- CI 配置与依赖分层：workflow 必须与核心测试入口保持一致 ----
    wf = BASE / '.github' / 'workflows' / 'ci.yml'
    wf_text = wf.read_text(encoding='utf-8') if wf.exists() else ''
    check('CI:workflow 引用核心测试、复现脚本与核心依赖',
          all(s in wf_text for s in ('test_core.py', 'reproduce.py',
                                     'requirements-core.txt')),
          f'{len(wf_text)} 字符')
    core_req = BASE / 'requirements-core.txt'
    req_text = core_req.read_text(encoding='utf-8') if core_req.exists() else ''
    # 只看真实依赖声明行：注释里「不含 tensorflow」这类说明文字不算依赖
    deps = [line.split('#', 1)[0].strip() for line in req_text.splitlines()]
    deps = [d for d in deps if d]
    check('CI:核心依赖不含 TensorFlow / sklearn',
          bool(deps)
          and any(d.startswith('numpy') for d in deps)
          and any(d.startswith('scipy') for d in deps)
          and not any(s in ' '.join(deps).lower()
                      for s in ('tensorflow', 'keras-tuner', 'scikit-learn')),
          '依赖=' + ', '.join(deps))

    import subprocess
    proc = subprocess.run([sys.executable, str(BASE / 'test_core.py'), '--json'],
                          capture_output=True, text=True, timeout=180,
                          encoding='utf-8', errors='replace')
    core_ok, core_total = False, 0
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            core_ok, core_total = payload['passed'], payload['total']
        except Exception:
            pass
    check('CI:test_core.py 独立运行全通过', core_ok,
          f'退出码 {proc.returncode}，{core_total} 项')

    fails = [n for n, ok2 in RESULTS if not ok2]
    print()
    print(f'共 {len(RESULTS)} 项，通过 {len(RESULTS) - len(fails)}，失败 {len(fails)}')
    if fails:
        print('失败项:', fails)
        sys.exit(1)


if __name__ == '__main__':
    main()
