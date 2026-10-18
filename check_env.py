"""Script de verificación del entorno de ejecución (PyTorch, CUDA, VRAM y soporte bf16)."""

import platform
import sys


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("VERIFICACIÓN DE ENTORNO - LOOPED-MATH")
    print("=" * 60)
    print(f"Sistema Operativo : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Versión de Python : {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch no está instalado en este entorno.")
        sys.exit(1)

    print(f"Versión de PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible   : {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Número de GPUs    : {device_count}")
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / (1024 ** 3)
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            print(f"  [GPU {idx}] Nombre      : {props.name}")
            print(f"  [GPU {idx}] VRAM Total  : {vram_gb:.2f} GB")
            print(f"  [GPU {idx}] Compute Cap.: {props.major}.{props.minor}")
            print(f"  [GPU {idx}] Soporta bf16: {bf16_supported}")
    else:
        print("AVISO: No se detectó GPU con CUDA. Se ejecutará en CPU.")

    print("=" * 60)


if __name__ == "__main__":
    main()
