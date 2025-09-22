import torch
import matplotlib.pyplot as plt

def visualizeFPMResults(O, P, iter_num):
    fig = plt.figure(88, figsize=(12, 14))
    plt.clf()

    # Inverse Fourier transform to get o(x, y)
    o = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(O)))

    ax = plt.subplot(3, 2, 1)
    im = ax.imshow(torch.log1p(torch.abs(O)).cpu().numpy(), cmap='gray')
    ax.set_title("Object Spectrum — Amplitude of O(kx, ky)")
    ax.set_xlabel("kx"); ax.set_ylabel("ky"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(3, 2, 2)
    im = ax.imshow(torch.angle(O).cpu().numpy(), cmap='gray')
    ax.set_title("Object Spectrum — Phase of O(kx, ky)")
    ax.set_xlabel("kx"); ax.set_ylabel("ky"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(3, 2, 3)
    im = ax.imshow(torch.abs(o).cpu().numpy(), cmap='gray')
    ax.set_title("Object Field — Amplitude of o(x, y)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(3, 2, 4)
    im = ax.imshow(torch.angle(o).cpu().numpy(), cmap='gray')
    ax.set_title("Object Field — Phase of o(x, y)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(3, 2, 5)
    im = ax.imshow(torch.abs(P).cpu().numpy(), cmap='gray')
    ax.set_title("Pupil Function — Amplitude of P(kx, ky)")
    ax.set_xlabel("kx"); ax.set_ylabel("ky"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(3, 2, 6)
    im = ax.imshow(torch.angle(P).cpu().numpy(), cmap='gray')
    ax.set_title("Pupil Function — Phase of P(kx, ky)")
    ax.set_xlabel("kx"); ax.set_ylabel("ky"); ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Fourier / Real-Space / Pupil — Iteration {iter_num} ",
                 y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.draw()
    plt.pause(0.001)
