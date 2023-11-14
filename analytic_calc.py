import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid')

def f_1(x, e_0, a, c):
    return (1 - x) * (1 + a * x * e_0) / (1 + c * x)

def lambda_1(x, e_0, c):
    return 1 / e_0 / (1 + c * x)

def f_2(x, e_0, a, b, c):
    return (1 - x) * e_0 * (1 + a * np.exp(b * x * e_0) - a) / (e_0 + c * np.exp(b * x * e_0) - c)

def lambda_2(x, e_0, b, c):
    return 1 / (e_0 + c * np.exp(b * x * e_0) - c)

def optimal_p_f():
    p = np.linspace(0, 1, 1001)

    p_1_ls = []
    f_1_ls = []
    p_2_ls = []
    f_2_ls = []
    pe0_1_ls = []
    pe0_2_ls = []

    e_0_ls = np.linspace(10, 100, 91)

    for e_0 in e_0_ls:
        func_1 = f_1(p, e_0, a, c)
        func_2 = f_2(p, e_0, a, b, c)
        p_1_ls.append(p[np.argmax(func_1)])
        pe0_1_ls.append(p[np.argmax(func_1)] * e_0)
        f_1_ls.append(np.max(func_1))
        p_2_ls.append(p[np.argmax(func_2)])
        pe0_2_ls.append(p[np.argmax(func_2)] * e_0)
        f_2_ls.append(np.max(func_2))

    plt.figure()
    ax= plt.subplot(111)
    ax.plot(e_0_ls, p_1_ls, linestyle='none', marker='o', label=r"$p_1$")
    ax.plot(e_0_ls, p_2_ls, linestyle='none', marker='o', label=r"$p_2$")
    ax.set_xlabel(r"$e_0$", fontsize = 24)
    ax.set_ylabel(r"$p$", fontsize = 24)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/c{round(c * 1000)}pm_a{round(a * 1000)}pm_p_e0.pdf")
    plt.close('all')

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(e_0_ls, pe0_1_ls, linestyle='none', marker='o', label=r"$p_1$")
    ax.plot(e_0_ls, pe0_2_ls, linestyle='none', marker='o', label=r"$p_2$")
    ax.set_xlabel(r"$e_0$", fontsize=24)
    ax.set_ylabel(r"$pe_0$", fontsize=24)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/c{round(c * 1000)}pm_a{round(a * 1000)}pm_pe0_e0.pdf")
    plt.close('all')

    plt.figure()
    ax= plt.subplot(111)
    ax.plot(e_0_ls, f_1_ls, linestyle='none', marker='o', label=r"$f_1$")
    ax.plot(e_0_ls, f_2_ls, linestyle='none', marker='o', label=r"$f_2$")
    ax.set_xlabel(r"$e_0$", fontsize = 24)
    ax.set_ylabel(r"$f$", fontsize = 24)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/c{round(c * 1000)}pm_a{round(a * 1000)}pm_f_e0.pdf")
    plt.close('all')

def optimal_lambda1():
    e_0 = 70
    p = np.linspace(0, 1, 1001)
    a, b, c = 0.1, 0.25, 3

    lambda_ls = []

    a_ls = np.logspace(-2, 0, 31)

    for a in a_ls:
        func_1 = f_1(p, e_0, a, c)
        p_1 = p[np.argmax(func_1)]
        lambda_ls.append(lambda_1(p_1, e_0, c))

    plt.figure()
    ax= plt.subplot(111)
    ax.plot(a_ls, lambda_ls, linestyle='none', marker='o')
    ax.set_xlabel(r"$\alpha$", fontsize = 24)
    ax.set_ylabel(r"$\lambda$", fontsize = 24)
    ax.set_xscale("log")
    ax.tick_params(labelsize=22)
    # ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/lambda_a1.pdf")
    plt.close('all')

    a, b, c = 0.1, 0.25, 3

    lambda_ls = []

    c_ls = np.logspace(-1, 1, 31)

    for c in c_ls:
        func_1 = f_1(p, e_0, a, c)
        p_1 = p[np.argmax(func_1)]
        lambda_ls.append(lambda_1(p_1, e_0, c))

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(a_ls, lambda_ls, linestyle='none', marker='o')
    ax.set_xlabel(r"$c$", fontsize=24)
    ax.set_ylabel(r"$\lambda$", fontsize=24)
    ax.set_xscale("log")
    ax.tick_params(labelsize=22)
    # ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/lambda_c1.pdf")
    plt.close('all')


def optimal_lambda2():
    e_0 = 70
    p = np.linspace(0, 1, 1001)
    a, b, c = 0.1, 0.25, 3

    lambda_ls = []

    a_ls = np.logspace(-2, 0, 31)

    for a in a_ls:
        func_2 = f_2(p, e_0, a, b, c)
        p_2 = p[np.argmax(func_2)]
        lambda_ls.append(lambda_2(p_2, e_0, b, c))

    plt.figure()
    ax= plt.subplot(111)
    ax.plot(a_ls, lambda_ls, linestyle='none', marker='o')
    ax.set_xlabel(r"$\alpha$", fontsize = 24)
    ax.set_ylabel(r"$\lambda$", fontsize = 24)
    ax.set_xscale("log")
    ax.tick_params(labelsize=22)
    # ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/lambda_a.pdf")
    plt.close('all')

    a, b, c = 0.1, 0.25, 3

    lambda_ls = []

    b_ls = np.logspace(-2, 0, 31)

    for b in b_ls:
        func_2 = f_2(p, e_0, a, b, c)
        p_2 = p[np.argmax(func_2)]
        lambda_ls.append(lambda_2(p_2, e_0, b, c))

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(b_ls, lambda_ls, linestyle='none', marker='o')
    ax.set_xlabel(r"$\beta$", fontsize=24)
    ax.set_ylabel(r"$\lambda$", fontsize=24)
    ax.set_xscale("log")
    ax.tick_params(labelsize=22)
    # ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/lambda_b.pdf")
    plt.close('all')

    a, b, c = 0.1, 0.25, 3

    lambda_ls = []

    c_ls = np.logspace(-2, 1, 31)

    for c in c_ls:
        func_2 = f_2(p, e_0, a, b, c)
        p_2 = p[np.argmax(func_2)]
        lambda_ls.append(lambda_2(p_2, e_0, b, c))

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(c_ls, lambda_ls, linestyle='none', marker='o')
    ax.set_xlabel(r"$c$", fontsize=24)
    ax.set_ylabel(r"$\lambda$", fontsize=24)
    ax.set_xscale("log")
    ax.tick_params(labelsize=22)
    # ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"figs_calc/lambda_c.pdf")
    plt.close('all')


b = 0.25
for a in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]:
    for c in [0.5, 1, 2, 3, 5, 10, 20, 30, 50]:
     optimal_p_f()

optimal_lambda2()
