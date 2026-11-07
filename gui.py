import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import audit
import evaluation
import randomness
import settlement
import simulation

# 每次更新数据后最多对账最近 N 份存档（更早的存档需用 settlement CLI 手动核对）。
SETTLE_MAX_FILES = 5
APP_VERSION = 'v2.30'


class LotteryPredictorGUI:
    def __init__(self, root, data_processor, lottery_model, save_dir):
        self.root = root
        self.data_processor = data_processor
        self.lottery_model = lottery_model
        self.save_dir = save_dir
        self.root.title(f"排列3 分析与选号工具 {APP_VERSION}")
        self.root.geometry("1000x820")
        self.root.configure(bg="#E0F7FA")
        self.num_rows = tk.StringVar(value="7000")
        self.fetch_periods = tk.StringVar(value="20")
        self.pick_count = tk.StringVar(value="5")
        self.pick_mode = tk.StringVar(value="直选")
        self.sim_mode = tk.StringVar(value="组选6")
        self.sim_bets = tk.StringVar(value="30")
        self.budget = tk.StringVar(value="100")
        self.update_queue = queue.Queue()
        self.is_running = False
        self.interrupt = False
        self._closed = False
        self.start_time = None
        self.progress_value = 0
        self._setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_queue)
        self.root.after(1000, self._tick)

    def _setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(self.main_frame, text="排列3 分析与选号工具", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10)

        input_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="10")
        input_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        ttk.Label(input_frame, text="样本数量：").grid(row=0, column=0)
        ttk.Entry(input_frame, textvariable=self.num_rows).grid(row=0, column=1)
        ttk.Label(input_frame, text="更新期数：").grid(row=1, column=0)
        ttk.Entry(input_frame, textvariable=self.fetch_periods).grid(row=1, column=1)
        ttk.Button(input_frame, text="初次运行", command=self.initial_run).grid(row=2, column=0)
        ttk.Button(input_frame, text="继续运行", command=self.continue_run).grid(row=2, column=1)
        self.interrupt_button = ttk.Button(input_frame, text="中断", command=self.interrupt_prediction,
                                           state="disabled")
        self.interrupt_button.grid(row=3, column=0)
        ttk.Button(input_frame, text="更新数据", command=self.fetch_and_update).grid(row=3, column=1)
        ttk.Button(input_frame, text="命中率回测(基线对照)",
                   command=self.backtest_run).grid(row=4, column=0, columnspan=2,
                                                  sticky="ew", pady=(6, 0))
        ttk.Label(input_frame, text="生成注数：").grid(row=5, column=0, pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.pick_count, width=8).grid(
            row=5, column=1, pady=(8, 0))
        ttk.Label(input_frame, text="玩法：").grid(row=6, column=0, pady=(6, 0))
        ttk.Combobox(input_frame, textvariable=self.pick_mode,
                     values=('直选', '组选6', '组选3'), state='readonly',
                     width=7).grid(row=6, column=1, pady=(6, 0))
        ttk.Button(input_frame, text="生成 N 注选号",
                   command=self.multi_pick_run).grid(row=7, column=0, columnspan=2,
                                                    sticky="ew", pady=(4, 0))
        ttk.Label(input_frame, text="模拟玩法：").grid(row=8, column=0, pady=(8, 0))
        ttk.Combobox(input_frame, textvariable=self.sim_mode,
                     values=('直选', '组选6', '组选3'), state='readonly',
                     width=7).grid(row=8, column=1, pady=(8, 0))
        ttk.Label(input_frame, text="模拟注数：").grid(row=9, column=0)
        ttk.Entry(input_frame, textvariable=self.sim_bets, width=8).grid(row=9, column=1)
        ttk.Button(input_frame, text="命中模拟(历史回放)",
                   command=self.simulate_run).grid(row=10, column=0, columnspan=2,
                                                  sticky="ew", pady=(4, 0))
        ttk.Label(input_frame, text="预算(元)：").grid(row=11, column=0, pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.budget, width=8).grid(row=11, column=1,
                                                                      pady=(8, 0))
        ttk.Button(input_frame, text="预算规划选号",
                   command=self.budget_run).grid(row=12, column=0, columnspan=2,
                                                sticky="ew", pady=(4, 0))
        ttk.Separator(input_frame, orient="horizontal").grid(
            row=13, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        ttk.Label(input_frame, text="严谨性检验",
                  font=("Arial", 9, "bold")).grid(row=14, column=0, columnspan=2,
                                                  sticky="w")
        ttk.Button(input_frame, text="累积审计（模型 vs 随机）",
                   command=self.audit_run).grid(row=15, column=0, columnspan=2,
                                                sticky="ew", pady=(4, 0))
        ttk.Button(input_frame, text="数据随机性检验",
                   command=self.randomness_run).grid(row=16, column=0, columnspan=2,
                                                     sticky="ew", pady=(4, 0))

        result_frame = ttk.Frame(self.main_frame)
        result_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure((1, 3, 5), weight=1)

        ttk.Label(result_frame, text="运行日志").grid(row=0, column=0)
        self.progress_text = tk.Text(result_frame, height=10, width=60)
        self.progress_text.grid(row=1, column=0, sticky="nsew")

        ttk.Label(result_frame, text="最近50个样本").grid(row=2, column=0)
        self.samples_tree = ttk.Treeview(
            result_frame, columns=("Period", "Numbers", "Group", "OddEven", "BigSmall", "Sum"),
            show="headings", height=10)
        self.samples_tree.grid(row=3, column=0, sticky="nsew")
        for col in ["Period", "Numbers", "Group", "OddEven", "BigSmall", "Sum"]:
            self.samples_tree.heading(col, text=col)
            self.samples_tree.column(col, width=80, anchor="center")

        ttk.Label(result_frame, text="运行结果").grid(row=4, column=0)
        self.result_text = tk.Text(result_frame, height=15, width=60)
        self.result_text.grid(row=5, column=0, sticky="nsew")

        self.progress = ttk.Progressbar(self.main_frame, length=600, mode='determinate', maximum=100)
        self.progress.grid(row=3, column=0, columnspan=2, pady=5)
        self.time_label = ttk.Label(self.main_frame, text="已运行时间：未开始")
        self.time_label.grid(row=4, column=0, columnspan=2)

    # ---- 主线程 UI ----
    def _on_close(self):
        self.interrupt = True      # 后台任务在下一个 epoch 检查点停止
        self._closed = True        # 之后 _log 直接丢弃，不再触碰已销毁的控件
        self.root.destroy()

    def _tick(self):
        if self._closed:
            return
        if self.is_running and self.start_time:
            mins, secs = divmod(int(time.time() - self.start_time), 60)
            self.time_label.configure(text=f"已运行时间：{mins}分 {secs}秒")
        self.root.after(1000, self._tick)

    def _process_queue(self):
        for _ in range(200):
            try:
                self.update_queue.get_nowait()()
            except queue.Empty:
                break
        self.root.after(100, self._process_queue)

    def _log(self, message):
        if self._closed:
            return
        self.update_queue.put(
            lambda: self.progress_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n"))

    def _update_progress(self):
        value = self.progress_value
        self.update_queue.put(lambda: self.progress.configure(value=value))

    def _set_running(self, running):
        self.is_running = running
        self.interrupt_button.config(state="normal" if running else "disabled")

    def _read_int(self, var, label):
        try:
            return int(var.get())
        except ValueError:
            messagebox.showwarning("输入有误", f"{label}必须是整数")
            return None

    def _start_worker(self, fn, *args):
        """启动一个 daemon 后台线程，绝不阻塞 Tk 主线程。"""
        if self.is_running:
            return
        self._set_running(True)
        self.interrupt = False
        self.progress_value = 0
        self.start_time = time.time()
        threading.Thread(target=self._safe_worker, args=(fn,) + args, daemon=True).start()

    def _safe_worker(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self._log(f"运行出错：{msg}")
            self.update_queue.put(lambda m=msg: messagebox.showerror("运行出错", m))
        finally:
            self.update_queue.put(lambda: self._set_running(False))

    # ---- 按钮回调（主线程）----
    def initial_run(self):
        num_rows = self._read_int(self.num_rows, "样本数量")
        if num_rows is not None:
            self._start_worker(self._run_prediction, True, num_rows)

    def continue_run(self):
        num_rows = self._read_int(self.num_rows, "样本数量")
        if num_rows is not None:
            self._start_worker(self._continue_prediction, num_rows)

    def fetch_and_update(self):
        periods = self._read_int(self.fetch_periods, "更新期数")
        if periods is not None:
            self._start_worker(self._do_fetch, periods)

    def backtest_run(self):
        self.progress_text.delete('1.0', tk.END)
        self._log("开始严格时序回测：预测第 t 期时只使用 data[:t]，不使用任何未来数据")
        self._start_worker(self._do_backtest)

    def audit_run(self):
        """汇总所有历史存档，检验模型选号在长期上是否优于随机。"""
        self.progress_text.delete('1.0', tk.END)
        self._log("开始累积审计：汇总全部 picks 存档的投注与真实命中，"
                  "按注级口径做二项检验")
        self._start_worker(self._do_audit)

    def randomness_run(self):
        """对开奖数据做九项卡方检验，判断是否存在可利用结构。"""
        self.progress_text.delete('1.0', tk.END)
        self._log("开始数据随机性检验：均匀性 / 三位独立 / 期际独立 / 形态 / 和值")
        self._start_worker(self._do_randomness)

    def _validated_pick(self):
        """主线程读取并校验 玩法+注数；合法返回 (count, mode)，否则 None。"""
        count = self._read_int(self.pick_count, "生成注数")
        if count is None:
            return None
        mode = self.pick_mode.get()
        rule = evaluation.PLAY_RULES.get(mode)
        if rule is None:
            messagebox.showwarning("输入有误", "玩法必须是 直选 / 组选6 / 组选3")
            return None
        if not 1 <= count <= rule['max_bets']:
            messagebox.showwarning("输入有误",
                                   f"{mode} 注数必须是 1~{rule['max_bets']} 的整数")
            return None
        return count, mode

    def multi_pick_run(self):
        validated = self._validated_pick()
        if validated:
            self._start_worker(self._do_multi_pick, *validated)

    def simulate_run(self):
        bets = self._read_int(self.sim_bets, "模拟注数")
        if bets is None:
            return
        mode = self.sim_mode.get()
        rule = evaluation.PLAY_RULES.get(mode)
        if rule is None:
            messagebox.showwarning("输入有误", "玩法必须是 直选 / 组选6 / 组选3")
            return
        if not 1 <= bets <= rule['max_bets']:
            messagebox.showwarning("输入有误",
                                   f"{mode} 注数必须是 1~{rule['max_bets']} 的整数")
            return
        self._log("开始命中模拟：随机不重复选号逐期回放（任何选法命中期望相同）")
        self._start_worker(self._do_simulate, mode, bets)

    def budget_run(self):
        budget = self._read_int(self.budget, "预算")
        if budget is None:
            return
        try:
            plan = evaluation.budget_plan(budget)
        except ValueError as e:
            messagebox.showwarning("输入有误", str(e))
            return
        bets_total = plan['bets']
        lines = [f"\n【预算规划】{budget} 元 = {bets_total} 注",
                 "玩法 | 单期中奖概率 | 成本 | 期望回报 | 返奖率"]
        for m, o in plan['modes'].items():
            lines.append(f"  {m} | {o['coverage_pct']:.2f}%（约1/{o['one_in']:.0f}）"
                         f" | {o['cost']:.0f} 元 | {o['expected_return']:.2f} 元"
                         f" | {o['return_rate'] * 100:.0f}%")
        lines.append("三种玩法返奖率相同（约52%）：预算规划提升的是中奖频率，"
                     "不改变亏损期望。")
        text = '\n'.join(lines) + '\n'
        self.update_queue.put(lambda: self.result_text.insert(tk.END, text))
        mode = self.pick_mode.get()
        rule = evaluation.PLAY_RULES[mode]
        bets = min(bets_total, rule['max_bets'])
        if bets_total > rule['max_bets']:
            self._log(f"预算可买 {bets_total} 注，超出{mode}上限 "
                      f"{rule['max_bets']} 注，仅投 {bets} 注")
        self._log(f"按当前玩法 {mode} 生成 {bets} 注选号（成本 {bets * 2} 元）")
        self._start_worker(self._do_multi_pick, bets, mode)

    def interrupt_prediction(self):
        self.interrupt = True
        self._log("已请求中断，将在当前 epoch 结束后停止训练")

    # ---- 后台任务 ----
    def _do_fetch(self, periods):
        self.data_processor.fetch_and_update(periods, self._log)
        self._display_samples()
        self._settle_pending()

    def _settle_pending(self):
        """更新数据后自动对账：核对存档选号在最新开奖上的真实命中。"""
        files = sorted(settlement.PICKS_DIR.glob('picks_*.json'))
        if not files:
            return
        self._log(f"发现 {len(files)} 份存档选号，开始对账...")
        try:
            import pandas as pd
            draws_df = pd.read_excel(self.data_processor.file_path, engine='openpyxl')
        except Exception as e:
            self._log(f"读取开奖数据失败，跳过对账：{e}")
            return
        for f in files[-SETTLE_MAX_FILES:]:
            try:
                report = settlement.settle(str(f),
                                           excel_path=self.data_processor.file_path,
                                           df=draws_df)
                self._log('---- ' + settlement.format_settlement(report))
            except Exception as e:
                self._log(f"对账失败 {f.name}：{e}")

    def _loaded_draw_data(self):
        """优先复用 DataProcessor 已加载数据，避免重复读 Excel；未加载才回退。"""
        if self.data_processor.data is not None:
            return self.data_processor.data
        return evaluation.load_excel_data()

    def _do_simulate(self, mode, bets):
        """命中模拟：随机不重复选号逐期回放（后台纯计算，无 Tk 依赖）。"""
        data = self._loaded_draw_data()
        start = max(60, len(data) // 2)
        self._log(f"读取 {len(data)} 期，模拟区间 [{start}, {len(data)})")
        result = simulation.simulate_random(data, mode, bets, start=start,
                                            interrupt=lambda: self.interrupt)
        for line in simulation.format_simulation_report(result).splitlines():
            self._log(line)
        self.update_queue.put(lambda: self.result_text.insert(
            tk.END, f"\n【命中模拟】{mode}×{result['bets']}注×{result['n']}期："
                    f"命中 {result['hit_draws']} 期（理论期望 "
                    f"{result['expected_hits']:.1f}），净损益 {result['pnl']:+.0f} 元\n"))
        self.progress_value = 100
        self._update_progress()

    def _do_backtest(self):
        """后台跑全量时序回测；UI 更新走队列，异常交 _safe_worker 统一弹窗。"""
        data = self._loaded_draw_data()
        start = max(60, len(data) // 2)
        self._log(f"读取 {len(data)} 期，评估区间 [{start}, {len(data)})")
        results = evaluation.run_backtest(data, start=start)
        report = evaluation.format_report(results, title='时序回测结果')
        for line in report.splitlines():
            self._log(line)
        n = next(iter(results.values()))['n']
        alpha = evaluation.bonferroni_threshold(results)
        sig = [name for name, r in results.items() if r['digit_p'] < alpha]
        summary = (f"\n【回测结论】{n} 期严格时序对照 | 直选随机期望 "
                   f"{n * evaluation.P_EXACT:.1f} 次 | 校正后显著优于随机的策略："
                   f"{('、'.join(sig)) if sig else '无'}\n")
        self.update_queue.put(lambda: self.result_text.insert(tk.END, summary))
        self.progress_value = 100
        self._update_progress()
        self._log("回测完成，完整报告见上方运行日志")

    def _do_audit(self):
        """后台执行累积审计：读取全部存档 → 对账 → 注级统计检验。"""
        reports, errors = audit.collect()
        summary = audit.summarize(reports)
        report = audit.format_audit(summary, errors, len(reports))
        for line in report.splitlines():
            self._log(line)
        for e in errors:
            self._log(f"  存档读取失败：{e}")
        n_arch = len(reports)
        verdict = "暂无可核对存档" if not summary else "已生成检验结论"
        self.update_queue.put(lambda: self.result_text.insert(
            tk.END, f"\n【累积审计】共 {n_arch} 份存档，{verdict}，"
                    f"详见上方运行日志\n"))
        self.progress_value = 100
        self._update_progress()

    def _do_randomness(self):
        """后台执行数据随机性检验。"""
        data = self._loaded_draw_data()
        result = randomness.analyze(data)
        report = randomness.format_report(result)
        for line in report.splitlines():
            self._log(line)
        n_holm = result.get('n_holm', 0)
        n_nominal = result.get('n_nominal', 0)
        verdict = ("全部符合随机（Holm 校正后）" if n_holm == 0
                   else f"Holm 校正后仍有 {n_holm} 项显著")
        extra = ("" if n_nominal == 0 or n_holm
                 else f"；裸判有 {n_nominal} 项 p<0.05，属多重比较波动，已校正")
        self.update_queue.put(lambda: self.result_text.insert(
            tk.END, f"\n【随机性检验】{result['n']} 期数据，{verdict}{extra}。"
                    f"若整体符合随机，则任何选号方式的命中率都是 k/1000\n"))
        self.progress_value = 100
        self._update_progress()

    def _do_multi_pick(self, count, mode):
        """生成 N 注互不重复的选号，并把理论命中率与成本如实写进结果区。

        mode 由主线程回调传入（后台线程不可读取 Tk 变量）。
        """
        if self.lottery_model.model is None:
            self._log("尚无可用模型，尝试加载上次状态...")
            self.data_processor.load_state(self.save_dir, self._log)
            self.lottery_model.load_state(self.save_dir, self._log)
        if self.lottery_model.model is None or self.data_processor.data is None:
            self.update_queue.put(lambda: messagebox.showwarning(
                "模型未就绪", "请先点「初次运行」或「继续运行」完成训练/加载后再生成选号"))
            return
        picks = self.lottery_model.predict_multiple(count, self._log, mode)
        if not picks:
            return
        try:
            path = settlement.save_picks(self.data_processor.last_period, mode, picks)
            self._log(f"选号已存档，更新数据后自动对账：{path.name}")
        except Exception as e:
            self._log(f"选号存档失败：{e}")
        odds = evaluation.play_odds(mode, count)
        per_bet_hit = odds['coverage'] / count          # 单注真实中奖概率
        per_bet_pct = per_bet_hit * 100
        lines = [f"\n【{count} 注{mode}】互不重复的组合（顺序含模型打分，无预测含义）"]
        for i, pick in enumerate(picks, 1):
            lines.append(f"  {i:>2}. {pick['numbers']}   模型打分 {pick['score']:.5f}")
        lines.append(f"注：模型打分只用来决定「从全部组合里挑这 {count} 注」，不是中奖概率。"
                     f"每注真实中奖概率恒为 {per_bet_pct:.2f}%（1/{1 / per_bet_hit:.0f}），"
                     f"与打分高低无关。")
        lines.append(f"单期中奖概率 {odds['coverage_pct']:.2f}%（约 1/{odds['one_in']:.0f}），"
                     f"与随机选 {count} 注完全相同")
        lines.append(f"按 2 元/注：成本 {odds['cost']:.0f} 元；单注奖金 {odds['payout']:.0f} 元，"
                     f"期望回报 {odds['expected_return']:.2f} 元"
                     f"（返奖率 {odds['return_rate'] * 100:.0f}%，长期期望亏 "
                     f"{abs(odds['expected_profit']):.2f} 元/期）\n")
        text = '\n'.join(lines)
        self.update_queue.put(lambda: self.result_text.insert(tk.END, text))
        self.progress_value = 100
        self._update_progress()
        self._log(f"{count} 注{mode}已生成，单期中奖概率 {odds['coverage_pct']:.2f}%")

    def _display_samples(self):
        if self.data_processor.recent_samples is None:
            return
        self.update_queue.put(lambda: self.samples_tree.delete(*self.samples_tree.get_children()))
        rows = list(self.data_processor.recent_samples.to_dict('records'))
        for row in rows:
            self.update_queue.put(lambda r=row: self.samples_tree.insert(
                "", "end", values=(r['Period'], r['Numbers'], r['Group'],
                                   r['OddEven'], r['BigSmall'], r['Sum'])))

    def _train_progress(self, frac):
        """后台训练线程推来的整体进度 0~1 → 进度条 20~80 区间（队列回主线程）。"""
        self.progress_value = 20 + int(60 * max(0.0, min(1.0, frac)))
        self._update_progress()

    def _run_prediction(self, is_initial, num_rows):
        self._log("开始运行..." if is_initial else "继续运行...")
        if is_initial:
            if not self.data_processor.load_data(num_rows, self._log):
                return
            self.progress_value = 20
            self._update_progress()
            if not self.lottery_model.train(self._log, lambda: self.interrupt,
                                            self._train_progress):
                self._log("训练未完成")
                return
            self.progress_value = 80
            self._update_progress()
        self._finish_prediction()

    def _continue_prediction(self, num_rows):
        self._log("正在加载上次状态...")
        if not (self.data_processor.load_state(self.save_dir, self._log)
                and self.lottery_model.load_state(self.save_dir, self._log)):
            self._log("未找到可用状态，转为初次运行")
            self._run_prediction(True, num_rows)
            return
        self.progress_value = 40
        self._update_progress()
        self._log("状态加载完成")
        self._finish_prediction()

    def _finish_prediction(self):
        """训练/加载完成后的结论块：故意不输出「下一期号码」。

        本数据集经 randomness 的九项检验与「独立均匀随机」相容，模型样本外
        命中率不超过随机基线；任何具体的三位数都会被读成「预测」，因此只给
        结论与下一步入口，号码一律走「生成 N 注」并附带真实概率与期望亏损。
        """
        self.data_processor.save_state(self.save_dir, self._log)
        self.lottery_model.save_state(self.save_dir, self._log)
        self.progress_value = 100
        self._update_progress()

        data = self.data_processor.data
        periods = len(data) if data is not None else 0
        text = (
            "\n【运行完成：模型已就绪，但不给出「下一期号码」】\n"
            f"  已使用 {periods} 期历史数据完成训练/状态加载。\n"
            "  本数据集经九项随机性检验（各位均匀、三位独立、期际转移独立等）\n"
            "  与「独立均匀随机」相容；模型在样本外不超过随机基线，训练集上的\n"
            "  高命中率来自记住历史样本。既然没有可学的结构，任何具体号码都\n"
            "  只能等价于随机选号，故此处不再输出单点「预测」。\n"
            "\n  要号码：点「生成 N 注」，会同时给出该方案的真实中奖概率、成本与期望亏损\n"
            "  要验证：点「命中率回测(基线对照)」，看模型是否真的强于随机\n"
            "  要体检：点「数据随机性检验」或「累积审计（模型 vs 随机）」\n")
        self.update_queue.put(lambda t=text: self.result_text.insert(tk.END, t))
        self.update_queue.put(lambda: messagebox.showinfo(
            "完成", "运行已完成（本工具不提供预测号码，选号请用「生成 N 注」）"))
