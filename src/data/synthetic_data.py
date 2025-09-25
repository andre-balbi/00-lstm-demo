import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=1000, sequence_length=30):
    print("Gerando dados sintéticos de chuva-vazão...")

    days = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 365) * 50 + 100

    precipitation = []
    prec = 50
    for i in range(n_samples):
        prec = 0.7 * prec + np.random.normal(0, 20) + seasonal[i] * 0.3
        precipitation.append(max(0, prec))

    volume = []
    vol = 1000
    for i in range(n_samples):
        if i > 0:
            vol = 0.9 * vol + 0.8 * precipitation[max(0, i-3)] - 10 + np.random.normal(0, 50)
        volume.append(max(500, vol))

    flow = []
    for i in range(n_samples):
        base_flow = 0.3 * precipitation[i] + 0.01 * volume[i]

        if precipitation[i] > 80:
            base_flow *= 1.5

        if i > 0:
            base_flow = 0.6 * base_flow + 0.4 * flow[i-1]

        flow.append(base_flow + np.random.normal(0, 5))

    data = pd.DataFrame({
        'data': days,
        'precipitacao': precipitation,
        'volume_reservatorio': volume,
        'vazao': flow
    })

    print(f"Dados gerados: {len(data)} amostras")
    print(f"Estatísticas:")
    print(f"    Precipitação média: {np.mean(precipitation):.1f} mm")
    print(f"    Volume médio: {np.mean(volume):.0f} m³")
    print(f"    Vazão média: {np.mean(flow):.1f} m³/s")

    return data