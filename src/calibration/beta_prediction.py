from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

class BetaPredictor:
    def __init__(self, trained_models_dir: str | Path = "trained_models") -> None:
        self._models_dir = Path(trained_models_dir)
        self._trained: dict = {}

    def predict(
        self,
        method: str,
        observed_beta: pd.Series,
        n_ahead: int = 1,
        **kwargs,
    ) -> pd.Series:
        if method == "last_value":
            return self._last_value(observed_beta, n_ahead)
        elif method == "rolling_mean":
            return self._rolling_mean(observed_beta, n_ahead, **kwargs)
        elif method == "expanding_mean":
            return self._expanding_mean(observed_beta, n_ahead)
        elif method == "biexponential":
            return self._biexponential(observed_beta, n_ahead)
        elif method == "median_beta":
            return self._median_beta(observed_beta, n_ahead)
        elif method == "regression_day":
            return self._regression_day(observed_beta, n_ahead)
        elif method == "regression_day_seir_prev_i":
            return self._regression_day_seir_prev_i(observed_beta, n_ahead, **kwargs)
        elif method == "shifted_last_value":
            return self._shifted_last_value(observed_beta, n_ahead, **kwargs)
        elif method == "shifted_rolling_mean":
            return self._shifted_rolling_mean(observed_beta, n_ahead, **kwargs)
        elif method == "incremental_last_value":
            return self._incremental_last_value(observed_beta, n_ahead)
        elif method == "incremental_rolling_mean":
            return self._incremental_rolling_mean(observed_beta, n_ahead, **kwargs)
        elif method == "lstm_day_e_prev_i":
            return self.lstm_day_e_prev_i(observed_beta, n_ahead, **kwargs)
        else:
            raise ValueError(f"unknown beta prediction method: {method}")

    def _last_value(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        last = beta.iloc[-1] if len(beta) > 0 else 0.0
        idx = range(len(beta), len(beta) + n_ahead)
        return pd.Series([last] * n_ahead, index=idx)

    def _rolling_mean(self, beta: pd.Series, n_ahead: int, window: int = 7) -> pd.Series:
        tail = beta.iloc[-window:] if len(beta) >= window else beta
        val = float(tail.mean()) if len(tail) > 0 else 0.0
        idx = range(len(beta), len(beta) + n_ahead)
        return pd.Series([val] * n_ahead, index=idx)

    def _expanding_mean(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        val = float(beta.mean()) if len(beta) > 0 else 0.0
        idx = range(len(beta), len(beta) + n_ahead)
        return pd.Series([val] * n_ahead, index=idx)

    def _biexponential(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        from scipy.optimize import curve_fit

        t = np.arange(len(beta), dtype=float)
        y = beta.values.astype(float)

        if len(y) < 3:
            return self._last_value(beta, n_ahead)

        def _model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a + b * np.exp(-c * t)

        try:
            popt, _ = curve_fit(
                _model, t, y,
                p0=[y[-1], y[0], 0.1],
                maxfev=5000,
            )
        except RuntimeError:
            return self._last_value(beta, n_ahead)

        future_t = np.arange(len(beta), len(beta) + n_ahead, dtype=float)
        preds = _model(future_t, *popt)
        preds = np.clip(preds, 0, y.max() * 2 if y.max() > 0 else 1.0)
        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def _median_beta(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        val = float(beta.median()) if len(beta) > 0 else 0.0
        idx = range(len(beta), len(beta) + n_ahead)
        return pd.Series([val] * n_ahead, index=idx)

    def _regression_day(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        if len(beta) < 2:
            return self._last_value(beta, n_ahead)

        t = np.arange(len(beta), dtype=float)
        coeffs = np.polyfit(t, beta.values.astype(float), 1)
        future_t = np.arange(len(beta), len(beta) + n_ahead, dtype=float)
        preds = np.polyval(coeffs, future_t)
        preds = np.maximum(preds, 0)
        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def _regression_day_seir_prev_i(
        self,
        beta: pd.Series,
        n_ahead: int,
        prev_i: pd.Series | None = None,
    ) -> pd.Series:
        if prev_i is None or len(beta) < 3:
            return self._regression_day(beta, n_ahead)

        t = np.arange(len(beta), dtype=float)
        x = np.column_stack([t, prev_i.values[:len(beta)].astype(float)])
        y = beta.values.astype(float)

        x_aug = np.column_stack([x, np.ones(len(x))])
        try:
            w, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return self._regression_day(beta, n_ahead)

        last_i = float(prev_i.iloc[-1]) if len(prev_i) > 0 else 0.0
        future_t = np.arange(len(beta), len(beta) + n_ahead, dtype=float)
        x_pred = np.column_stack([future_t, np.full(n_ahead, last_i), np.ones(n_ahead)])
        preds = x_pred @ w
        preds = np.maximum(preds, 0)
        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def _shifted_last_value(
        self, beta: pd.Series, n_ahead: int, shift: int = 7,
    ) -> pd.Series:
        idx_src = max(0, len(beta) - shift)
        val = float(beta.iloc[idx_src]) if len(beta) > 0 else 0.0
        return pd.Series([val] * n_ahead, index=range(len(beta), len(beta) + n_ahead))

    def _shifted_rolling_mean(
        self, beta: pd.Series, n_ahead: int, shift: int = 7, window: int = 7,
    ) -> pd.Series:
        end_idx = max(0, len(beta) - shift)
        start_idx = max(0, end_idx - window)
        segment = beta.iloc[start_idx:end_idx]
        val = float(segment.mean()) if len(segment) > 0 else 0.0
        return pd.Series([val] * n_ahead, index=range(len(beta), len(beta) + n_ahead))

    def _incremental_last_value(self, beta: pd.Series, n_ahead: int) -> pd.Series:
        if len(beta) < 2:
            return self._last_value(beta, n_ahead)
        delta = float(beta.iloc[-1] - beta.iloc[-2])
        preds = [float(beta.iloc[-1]) + delta * (i + 1) for i in range(n_ahead)]
        preds = [max(0, p) for p in preds]
        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def _incremental_rolling_mean(
        self, beta: pd.Series, n_ahead: int, window: int = 7,
    ) -> pd.Series:
        if len(beta) < window + 1:
            return self._rolling_mean(beta, n_ahead, window)
        recent = beta.iloc[-window:]
        prev = beta.iloc[-window - 1:-1]
        delta = float(recent.mean() - prev.mean())
        base = float(recent.mean())
        preds = [max(0, base + delta * (i + 1)) for i in range(n_ahead)]
        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def lstm_day_e_prev_i(
        self, beta: pd.Series, n_ahead: int, prev_i: pd.Series | None = None,
    ) -> pd.Series:
        model_path = self._models_dir / "lstm_day_E_prev_I_for_seir.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                "trained model not found at trained_models/lstm_day_E_prev_I_for_seir.keras, "
                "run training notebook from Network_hybrid first"
            )

        if "lstm_day_e_prev_i" not in self._trained:
            from tensorflow import keras
            self._trained["lstm_day_e_prev_i"] = keras.models.load_model(str(model_path))
            logger.info("loaded LSTM model from {}", model_path)

        model = self._trained["lstm_day_e_prev_i"]

        t = np.arange(len(beta), dtype=float)
        e_vals = beta.values.astype(float)
        i_vals = prev_i.values.astype(float) if prev_i is not None else np.zeros(len(beta))

        features = np.column_stack([t, e_vals, i_vals])

        window = model.input_shape[1] if len(model.input_shape) >= 2 else len(beta)
        preds = []
        current = features.copy()
        for step in range(n_ahead):
            seq = current[-window:].reshape(1, window, features.shape[1])
            y_hat = float(model.predict(seq, verbose=0).flat[0])
            y_hat = max(y_hat, 0.0)
            preds.append(y_hat)
            next_row = np.array([len(beta) + step, y_hat, i_vals[-1] if len(i_vals) else 0.0])
            current = np.vstack([current, next_row])

        return pd.Series(preds, index=range(len(beta), len(beta) + n_ahead))

    def load_trained_model(self, method: str, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning("trained model not found: {}", path)
            return

        if path.suffix == ".joblib":
            import joblib
            self._trained[method] = joblib.load(path)
        elif path.suffix in (".keras", ".h5"):
            from tensorflow import keras
            self._trained[method] = keras.models.load_model(str(path))
        else:
            logger.warning("unsupported model format: {}", path.suffix)

        logger.info("loaded trained model for {}: {}", method, path)
