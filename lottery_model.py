import itertools
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path

import keras_tuner as kt
import keras_tuner.errors as keras_tuner_errors
import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.callbacks import (EarlyStopping, LambdaCallback,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling1D, Input, Lambda,
                                     Layer, LayerNormalization, MultiHeadAttention, Reshape,
                                     Softmax)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2


BASE_DIR = Path(__file__).resolve().parent
# 三个位置各预测 0-9，覆盖全部 1000 种有序且可重复的开奖结果。
OUTPUT_SHAPE = (3, 10)
# 默认输入序列长度（config 无该项时的回退值；GUI 与 CLI 同值）。
SEQUENCE_LENGTH_DEFAULT = 50
# 加载 h5 时对杀软短暂独占文件的重试参数。
LOAD_READ_ATTEMPTS = 4
LOAD_RETRY_WAIT = 2.0
# 各玩法的组合空间与单注中奖概率（与 evaluation.PLAY_RULES 的数字一致）。
COMBO_MAX_BETS = {'直选': 1000, '组选6': 120, '组选3': 90}
COMBO_PER_BET = {'直选': 1, '组选6': 6, '组选3': 3}


def sliding_windows(matrix, window):
    """(N, D) → (N-window+1, window, D) 滑窗数组。

    sliding_window_view 是零拷贝视图，ascontiguousarray 只做一次内存复制，
    等价并取代逐窗列表切片复制（训练集上万窗口时明显更快）。
    """
    view = sliding_window_view(matrix, (window, matrix.shape[1]))[:, 0]
    return np.ascontiguousarray(view)


def iter_combos(mode):
    """按玩法枚举全部投注组合字符串（rank_combos / 模拟器抽样 / 全收共用）。

    直选：1000 个有序三位组合；组选6：三个互不相同数字的无序组（120 个）；
    组选3：恰有一对相同数字的无序组（90 个）。未知玩法抛 ValueError。
    """
    if mode == '直选':
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    yield f'{a}{b}{c}'
    elif mode in ('组选6', '组选3'):
        for a in range(10):
            for b in range(a, 10):
                for c in range(b, 10):
                    distinct = len({a, b, c})
                    if (mode == '组选6' and distinct == 3) or (mode == '组选3' and distinct == 2):
                        yield f'{a}{b}{c}'
    else:
        raise ValueError(f'未知玩法：{mode}')


def rank_combos(probs, mode, k):
    """按模型概率降序给出 k 注互不重复组合（predict_multiple 与模拟器共用）。

    probs 形状 (3, 10)；组选按无序数字组枚举，score 为该组任一排列
    的概率之和，即该注的中奖概率（模型口径）。
    """
    if mode not in COMBO_MAX_BETS:
        raise ValueError(f'未知玩法：{mode}')
    k = int(k)
    if not 1 <= k <= COMBO_MAX_BETS[mode]:
        raise ValueError(f'{mode} 注数须在 1~{COMBO_MAX_BETS[mode]} 之间')

    def ordered_score(x, y, z):
        return float(probs[0, x] * probs[1, y] * probs[2, z])

    combos = []
    for s in iter_combos(mode):
        digits = (int(s[0]), int(s[1]), int(s[2]))
        if mode == '直选':
            score = ordered_score(*digits)
        else:
            score = sum(ordered_score(*p) for p in set(itertools.permutations(digits)))
        combos.append((float(score), s))
    combos.sort(key=lambda item: (-item[0], item[1]))
    return [{'numbers': num, 'score': score} for score, num in combos[:k]]


class TransformerEncoderLayer(Layer):
    def __init__(self, units, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)
        self.ffn = tf.keras.Sequential([Dense(units * 4, activation='relu', kernel_regularizer=l2(1e-5)),
                                        Dense(units * 2, activation='relu', kernel_regularizer=l2(1e-5)),
                                        Dense(units, kernel_regularizer=l2(1e-5))])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'num_heads': self.num_heads, 'dropout': self.dropout})
        return config


def focal_loss(gamma=2.0, alpha=0.25):
    """保留仅为兼容旧的 saved_state；当前模型改用 sparse categorical crossentropy。"""
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred) - \
               (1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss)

    return focal_loss_fn


class _InterruptedError(keras_tuner_errors.FatalError):
    """训练被用户中断。

    keras-tuner 会吞掉 trial 内的普通异常（标记 INVALID 继续搜索），
    只有 FatalError 会向上抛出终止整个 search()，因此中断必须用它。
    """


class _InterruptCallback(tf.keras.callbacks.Callback):
    """epoch 级响应中断标志。

    raise_on_interrupt=True 用于 tuner.search（抛 _InterruptedError 终止
    全部搜索）；False 用于普通 fit（置 stop_training 优雅停在当前 epoch）。
    """

    def __init__(self, flag, raise_on_interrupt=False):
        super().__init__()
        self.flag = flag
        self.raise_on_interrupt = raise_on_interrupt
        self.triggered = False

    def on_epoch_end(self, epoch, logs=None):
        if self.flag():
            self.triggered = True
            if self.raise_on_interrupt:
                raise _InterruptedError('用户中断训练')
            self.model.stop_training = True


class _ProgressHyperband(kt.Hyperband):
    """on_trial_end 时推搜索进度（Hyperband 典型约 16 个 trial，仅近似值）。"""

    def __init__(self, *args, progress_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_fn = progress_fn
        self._trials_done = 0

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        self._trials_done += 1
        if self._progress_fn:
            self._progress_fn(min(1.0, self._trials_done / 16.0))


class LotteryModel:
    def __init__(self, sequence_length, data_processor):
        self.sequence_length = sequence_length
        self.data_processor = data_processor
        self.model = None
        self.best_hparams = None

    def build_model(self, hp=None):
        if hp is None:
            units, dropout, lr, num_heads, num_layers = 256, 0.3, 0.0001, 4, 3
        else:
            units = hp.Int('units', 128, 256, step=64)
            dropout = hp.Float('dropout', 0.2, 0.4, step=0.1)
            lr = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
            num_heads = hp.Choice('num_heads', [4, 8])
            num_layers = hp.Choice('num_layers', [2, 3])

        inputs = Input(shape=(self.sequence_length, 9))
        match_weights = [hp.Float(f'match_weight_{i}', 0.5, 1.0, step=0.25) if hp else 0.75
                         for i in range(1, 4)]
        feature_weights = [hp.Float(f'feature_weight_{i}', 0.5, 1.5, step=0.25) if hp else 1.0
                           for i in range(1, 4)]
        weight_matrix = np.diag([1, 1, 1, *match_weights, *feature_weights]).astype(np.float32)
        x = Lambda(lambda t: tf.matmul(t, weight_matrix))(inputs)
        x = Dense(units, use_bias=False)(x)

        position = tf.range(self.sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, units, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / units))
        pos_encoding = tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)
        if pos_encoding.shape[1] > units:
            pos_encoding = pos_encoding[:, :units]
        elif pos_encoding.shape[1] < units:
            pos_encoding = tf.pad(pos_encoding, [[0, 0], [0, units - pos_encoding.shape[1]]])
        x = x + pos_encoding

        for _ in range(num_layers):
            x = TransformerEncoderLayer(units, num_heads, dropout)(x)
        x = GlobalAveragePooling1D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        x = Dense(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])(x)
        x = Reshape(OUTPUT_SHAPE)(x)
        outputs = Softmax(axis=-1)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='digit_acc')])
        return model

    def preprocess_data(self, log_callback, fit_scalers=True):
        """滑窗构造 (N-L, L, 9) 输入；标签为窗口之后那一期的 3 位数字 (N-L, 3)。"""
        dp = self.data_processor
        if dp.data is None or len(dp.data) < self.sequence_length + 1:
            log_callback(f"数据不足：需要至少 {self.sequence_length + 1} 条")
            return None, None
        matrix = dp.build_feature_matrix(fit_scalers=fit_scalers)
        L = self.sequence_length
        count = len(matrix) - L
        # 滑窗视图含 N-L+1 个窗口；标签 y[j]=data[j+L] 要求窗口后必须还有
        # 一期，因此截断到 count = N-L 个（与旧版逐窗列表完全一致）。
        X = sliding_windows(matrix, L)[:count]
        y = dp.data[L:L + count].astype(np.int32)
        log_callback(f"构造样本 {count} 条，输入 {X.shape}，标签 {y.shape}")
        return X, y

    def train(self, log_callback, interrupt_flag, progress_callback=None):
        """训练模型；interrupt_flag 每 epoch 结束检查一次，progress_callback
        收到 0~1 的整体训练进度（搜索阶段 0~0.8，最终训练 0.8~1.0）。"""
        X, y = self.preprocess_data(log_callback, fit_scalers=True)
        if X is None:
            return False
        # 时间序列必须按时间切分，不能随机 validation_split（会把未来泄进训练）。
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        log_callback(f"训练 {len(X_train)} 条 / 验证 {len(X_val)} 条（按时间顺序切分）")

        def _progress(frac):
            if progress_callback:
                try:
                    progress_callback(min(1.0, max(0.0, frac)))
                except Exception:
                    pass

        tuner_dir = str(BASE_DIR / 'tuner_dir')
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir)
        tuner = _ProgressHyperband(
            self.build_model, objective='val_loss', max_epochs=30, factor=3,
            directory=tuner_dir, project_name='lottery_predictor',
            progress_fn=lambda f: _progress(0.8 * f))
        search_cb = _InterruptCallback(interrupt_flag, raise_on_interrupt=True)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                     search_cb]
        try:
            tuner.search(X_train, y_train, epochs=30, batch_size=16,
                         validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        except _InterruptedError:
            log_callback("训练已中断")
            return False
        if interrupt_flag():
            log_callback("训练已中断")
            return False
        self.best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.get_best_models(num_models=1)[0]
        log_callback(f"最佳超参数：{self.best_hparams.values}")
        total_epochs = 50
        fit_cb = _InterruptCallback(interrupt_flag)

        def _fit_progress(epoch, logs=None):
            _progress(0.8 + 0.2 * (epoch + 1) / total_epochs)
            if (epoch + 1) % 10 == 0 or epoch + 1 == total_epochs:
                log_callback(f"最终训练进度 {epoch + 1}/{total_epochs} epochs")

        self.model.fit(X_train, y_train, epochs=total_epochs, batch_size=16,
                       validation_data=(X_val, y_val),
                       callbacks=callbacks[:2] + [fit_cb,
                                                  LambdaCallback(on_epoch_end=_fit_progress)],
                       verbose=0)
        if fit_cb.triggered:
            log_callback("训练已中断，模型未保存")
            return False
        return True

    def _next_probabilities(self, log_callback):
        """模型对下一期 3 个位置的概率分布，形状 (3, 10)。"""
        if self.model is None or self.data_processor.data is None:
            log_callback("模型或数据未准备好")
            return None
        L = self.sequence_length
        matrix = self.data_processor.build_feature_matrix(fit_scalers=False)
        if len(matrix) < L:
            log_callback("数据不足，无法构造预测窗口")
            return None
        input_data = matrix[-L:].reshape(1, L, 9)
        return self.model.predict(input_data, verbose=0)[0]   # (3, 10)

    def predict_next(self, log_callback):
        probs = self._next_probabilities(log_callback)
        if probs is None:
            return None
        digits = np.argmax(probs, axis=-1).astype(int).tolist()
        log_callback(f"三位数字各自最高概率取值：{digits}")
        return ''.join(map(str, digits))

    def predict_multiple(self, k, log_callback, mode='直选'):
        """按模型概率降序给出 k 注互不重复的投注组合。

        mode='直选'：有序三位数，单注中奖概率 1/1000；
        mode='组选6'：三个不同数字、顺序不限，单注 6/1000；
        mode='组选3'：恰有一对相同数字、顺序不限，单注 3/1000。
        概率只决定「选哪 k 注」，不改变命中率：只要 k 注互不重复，
        单期中奖概率 = k × 单注概率，与随机选注完全相同。

        返回 [{'numbers': '049', 'score': 0.0142}]，score 为该注中奖概率；
        模型/数据未就绪或参数非法时返回 None。
        """
        if mode not in COMBO_MAX_BETS:
            log_callback("玩法必须是 直选 / 组选6 / 组选3")
            return None
        probs = self._next_probabilities(log_callback)
        if probs is None:
            return None
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 0
        max_bets = COMBO_MAX_BETS[mode]
        if not 1 <= k <= max_bets:
            log_callback(f"{mode} 注数必须是 1~{max_bets} 的整数")
            return None
        picks = rank_combos(probs, mode, k)
        per_bet = COMBO_PER_BET[mode]
        log_callback(f"按模型概率生成 {k} 注不重复{mode}组合；"
                     f"单期中奖概率 {k * per_bet / 1000:.2%}（与随机选注相同）")
        return picks

    def save_state(self, save_dir, log_callback):
        """保存模型到全新唯一文件（model_时间戳_随机.h5）。

        本机环境下覆盖已存在的 h5 文件会被杀软/索引器间歇性独占
        （PermissionError / h5 内部重名错误），因此不覆盖：每次写全新
        文件，写完尝试清理旧文件，被占用则留给下次清理。
        """
        import glob
        os.makedirs(save_dir, exist_ok=True)
        if self.model:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(save_dir, f'model_{stamp}_{uuid.uuid4().hex[:6]}.h5')
            self.model.save(path)
            for old in glob.glob(os.path.join(save_dir, 'model_*.h5')):
                if os.path.abspath(old) != os.path.abspath(path):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
        with open(os.path.join(save_dir, 'model_state.pkl'), 'wb') as f:
            pickle.dump({'best_hparams': self.best_hparams,
                         'sequence_length': self.sequence_length,
                         'output_shape': OUTPUT_SHAPE}, f)
        log_callback("模型状态已保存")

    def load_state(self, save_dir, log_callback):
        import glob
        candidates = sorted(glob.glob(os.path.join(save_dir, 'model_*.h5')),
                            key=os.path.getmtime)
        if not candidates:
            return False
        src = candidates[-1]
        # 模型文件可能被杀软短暂独占：读入内存后经唯一临时文件加载，
        # Keras 从不直接打开原文件，也就不会长期持有它的句柄。
        try:
            for i in range(LOAD_READ_ATTEMPTS):
                try:
                    with open(src, 'rb') as f:
                        payload = f.read()
                    break
                except PermissionError:
                    if i == LOAD_READ_ATTEMPTS - 1:
                        log_callback('模型文件暂时被占用，请稍后重试')
                        return False
                    time.sleep(LOAD_RETRY_WAIT)
            tmpc = f'{src}.load_{uuid.uuid4().hex[:8]}.tmp'
            try:
                with open(tmpc, 'wb') as f:
                    f.write(payload)
                model = load_model(tmpc, custom_objects={
                    'TransformerEncoderLayer': TransformerEncoderLayer,
                    'focal_loss_fn': focal_loss()}, compile=False)
            finally:
                try:
                    os.remove(tmpc)
                except OSError:
                    pass
        except Exception as e:
            log_callback(f"加载模型状态失败：{str(e)}")
            return False
        shape = model.output_shape
        if not (isinstance(shape, tuple) and tuple(shape[1:]) == OUTPUT_SHAPE):
            log_callback("已有模型输出结构与当前版本不符，需重新训练")
            return False
        state_path = os.path.join(save_dir, 'model_state.pkl')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                if state.get('sequence_length') != self.sequence_length:
                    log_callback("序列长度与已保存模型不一致，需重新训练")
                    return False
                self.best_hparams = state.get('best_hparams')
            except Exception as e:
                log_callback(f"读取模型元信息失败：{str(e)}")
                return False
        self.model = model
        log_callback("已加载模型状态")
        return True
