"""Independent dense Chebyshev channel checks; no solver implementation imports."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
rng = np.random.default_rng(1729)
rows = []
for n in [2, 4]:
    for kappa in [2, 10, 100, 1000]:
        left = np.linalg.qr(rng.normal(size=(n, n)))[0]
        right = np.linalg.qr(rng.normal(size=(n, n)))[0]
        matrix = (left * np.geomspace(1, 1 / kappa, n)) @ right.T
        beta = rng.normal(size=n); beta /= np.linalg.norm(beta)
        gamma = 1 / kappa; gap = gamma**2 / (1 + gamma**2)
        dimension = 1 << n.bit_length()
        F = np.zeros((dimension, dimension))
        F[:n, :n] = matrix; F[:n, n] = -gamma * beta
        F[n+1:, n+1:] = np.eye(dimension-n-1)
        F /= np.sqrt(1 + gamma**2)
        G = F.T @ F
        values, basis = np.linalg.eigh(G)
        values[np.abs(values) < gap / 4] = 0
        r = np.eye(dimension)[:, n]; R = np.outer(r, r)
        target = np.r_[gamma * np.linalg.solve(matrix, beta), 1, np.zeros(dimension-n-1)]
        target /= np.linalg.norm(target); Pi = np.outer(target, target)
        q = float(target @ R @ target)
        default = int(np.ceil(np.arccosh(10) / (2 * np.arctanh(np.sqrt(gap)))))
        for s in [1, 2, 4, 8, 16, default]:
            # Independent three-term evaluation, with the kernel set analytically.
            x = (1 + np.longdouble(gap) - 2 * values.astype(np.longdouble)) / (1 - np.longdouble(gap))
            t0, t1 = np.ones_like(x), x.copy()
            for degree in range(2, s + 1): t0, t1 = t1, 2 * x * t1 - t0
            denominator = np.cosh(2 * s * np.arctanh(np.sqrt(np.longdouble(gap))))
            response = np.asarray(t1 / denominator, dtype=float); response[values == 0] = 1
            eta = float(1 / denominator)
            assert np.max(np.abs(response[values > 0])) <= eta + 2e-9
            B = (basis * response) @ basis.T; H = np.eye(dimension) - B @ B
            h, v = np.linalg.eigh(H); sqrtH = (v * np.sqrt(np.maximum(h, 0))) @ v.T
            kraus = [B] + [np.outer(r, np.eye(dimension)[:, j]) @ sqrtH for j in range(dimension)]
            completeness = float(np.linalg.norm(sum(K.T @ K for K in kraus) - np.eye(dimension), 2))
            choi = np.block([[B[:, i:i+1] @ B[:, j:j+1].T + H[j, i] * R for j in range(dimension)] for i in range(dimension)])
            minimum_choi = float(np.linalg.eigvalsh(choi)[0])
            output_target = B @ Pi @ B + np.trace(H @ Pi) * R
            stationary_error = float(np.linalg.norm(output_target - Pi, 2))
            max_slack = -float('inf'); max_attenuation_slack = -float('inf')
            for repeat in range(5):
                raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
                rho = raw @ raw.conj().T; rho /= np.trace(rho)
                before = 1 - np.trace(Pi @ rho).real
                after = 1 - np.trace(Pi @ (B @ rho @ B + np.trace(H @ rho) * R)).real
                max_slack = max(max_slack, float(after - (1 - q * (1 - eta**2)) * before))
                attenuation = 1 - 1e-6; Ba = attenuation * B; Ha = np.eye(dimension) - Ba @ Ba
                after_a = 1 - np.trace(Pi @ (Ba @ rho @ Ba + np.trace(Ha @ rho) * R)).real
                bound = attenuation**2 * (1 - q * (1 - eta**2)) * before + (1-q) * (1-attenuation**2)
                max_attenuation_slack = max(max_attenuation_slack, float(after_a - bound))
            assert completeness < 1e-10 and minimum_choi > -1e-10 and stationary_error < 1e-10
            assert max_slack < 1e-10 and max_attenuation_slack < 1e-10
            rows.append({'size':n, 'condition':kappa, 'degree':s, 'choi_minimum_eigenvalue':minimum_choi,
                         'kraus_completeness_error':completeness, 'stationary_error':stationary_error,
                         'contraction_excess':max_slack, 'attenuation_bound_excess':max_attenuation_slack})
slacks = []
for repeat in range(6000):
    q = rng.uniform(0, 1); gap = rng.uniform(0, 1); order = int(rng.integers(1, 65))
    slacks.append(1-q*(1-(1-gap)**order) - (1-q*gap)**order)
assert min(slacks) > -1e-13
payload = {'passed':True, 'independent_implementation':True, 'seed':1729,
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'cases':rows, 'general_q_convexity_points':len(slacks), 'minimum_convexity_slack':min(slacks),
           'interpretation':'Floating-point dense regression checks of analytic results, not interval proofs or physical evidence'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
print('PASS:', len(rows), 'independent Chebyshev cases and', len(slacks), 'general-q convexity points')
