"""
Generates various classes of states.
"""
import numpy as np

from scipy.special import binom
from math import sqrt


from qutip import Qobj
from qutip.states import fock_dm, thermal_dm, coherent_dm, coherent, basis, fock
from qutip.operators import displace
from qutip import Qobj
from qutip.random_objects import rand_dm


def cat(hilbert_size, alpha, S=0, mu=0):
    """
    Generates a cat state.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_

    
    Args:
    -----
        hilbert_size (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
                 default: 0
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state using the cat state.
                  default: 0


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state ket.
    """
    kend = 2 * S + 1
    cstates = 0 * (coherent(hilbert_size, 0))

    for k in range(0, int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(hilbert_size, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    ket = cstates.unit()
    return ket


def _get_num_prob(idx):
    """Selects a random probability vector from the list of number states"""
    states17 = [
        [
            (np.sqrt(7 - np.sqrt(17))) / np.sqrt(6),
            0,
            0,
            (np.sqrt(np.sqrt(17) - 1) / np.sqrt(6)),
            0,
        ],
        [
            0,
            (np.sqrt(9 - np.sqrt(17)) / np.sqrt(6)),
            0,
            0,
            (np.sqrt(np.sqrt(17) - 3) / np.sqrt(6)),
        ],
    ]

    statesM = [
        [
            0.5458351325482939,
            -3.7726009161224436e-9,
            4.849511177634774e-8,
            -0.7114411727633639,
            -7.48481181758003e-8,
            -1.3146003192319789e-8,
            0.44172510726665587,
            1.1545802803733896e-8,
            1.0609402576342428e-8,
            -0.028182506843720707,
            -6.0233214626778965e-9,
            -6.392041552216322e-9,
            0.00037641909140801935,
            -6.9186916801058116e-9,
        ],
        [
            2.48926815257019e-9,
            -0.7446851186077535,
            -8.040831059521339e-9,
            6.01942995399906e-8,
            -0.5706020908811399,
            -3.151900508005823e-8,
            -7.384935824733578e-10,
            -0.3460030551087218,
            -8.485651303145757e-9,
            -1.2114327561832047e-8,
            0.011798401879159238,
            -4.660460771433317e-9,
            -5.090374160706911e-9,
            -0.00010758601713550998,
        ],
    ]

    statesP = [
        [
            0.0,
            0.7562859301326029,
            0.0,
            0.0,
            -0.5151947804474741,
            -0.20807866860791188,
            0.12704803323656158,
            0.05101928893751686,
            0.3171198939841734,
        ],
        [
            -0.5583217426728544,
            -0.0020589109231194413,
            0.0,
            -0.7014041964402703,
            -0.05583041652626998,
            0.0005664728465725445,
            -0.2755044401850055,
            -0.3333309025086189,
            0.0785824556163142,
        ],
    ]

    statesP2 = [
        [
            -0.5046617350158988,
            0.08380989527942606,
            -0.225295417417812,
            0.0,
            -0.45359477373452817,
            -0.5236866813756252,
            0.2523308675079494,
            0.0,
            0.09562538828178244,
            0.2172849136874009,
            0.0,
            0.0,
            0.0,
            -0.2793663175980869,
            -0.08280858231312467,
            -0.05106696128137072,
        ],
        [
            -0.0014249418817930378,
            0.5018692341095683,
            0.4839749920101922,
            -0.3874886488913531,
            0.055390715144453026,
            -0.25780190053922486,
            -0.08970154713375252,
            -0.1892386424818236,
            0.10840637100094529,
            -0.19963901508324772,
            -0.41852779130900664,
            -0.05747247660559087,
            0.0,
            -0.0007888071131354318,
            -0.1424131123943283,
            -0.0001441905475623907,
        ],
    ]

    statesM2 = [
        [
            -0.45717455741713664,
            np.complex(-1.0856965103853774e-6, 1.3239037829080093e-6),
            np.complex(-0.35772784377291084, -0.048007740168066144),
            np.complex(-3.5459165445315755e-6, 0.000012571453643232864),
            np.complex(-0.5383420820794502, -0.24179040513272307),
            np.complex(9.675641330014822e-7, 4.569566899500361e-6),
            np.complex(0.2587482691377581, 0.313044506480362),
            np.complex(4.1979351791851435e-6, -1.122460690803522e-6),
            np.complex(-0.11094500303308243, 0.20905585817734396),
            np.complex(-1.1837814323046472e-6, 3.8758497675466054e-7),
            np.complex(0.1275629945870373, -0.1177987279989385),
            np.complex(-2.690647673469878e-6, -3.6519804939862998e-6),
            np.complex(0.12095531973074151, -0.19588735180644176),
            np.complex(-2.6588791126371675e-6, -6.058292629669095e-7),
            np.complex(0.052905370429015865, -0.0626791930782206),
            np.complex(-1.6615538648519722e-7, 6.756126951837809e-8),
            np.complex(0.016378329200891946, -0.034743342821208854),
            np.complex(4.408946495377283e-8, 2.2826415255126898e-8),
            np.complex(0.002765352838800482, -0.010624191776867055),
            6.429253878486627e-8,
            np.complex(0.00027095836439738105, -0.002684435917226972),
            np.complex(1.1081202749445256e-8, -2.938812506852636e-8),
            np.complex(-0.000055767533641099717, -0.000525444354381421),
            np.complex(-1.0776974926155464e-8, -2.497769263148397e-8),
            np.complex(-0.000024992489351114305, -0.00008178444317382933),
            np.complex(-1.5079116121444066e-8, -2.0513760149701907e-8),
            np.complex(-5.64035228941742e-6, -0.000010297667130821428),
            np.complex(-1.488452012610573e-8, -1.7358623165948514e-8),
            np.complex(-8.909884885392901e-7, -1.04267002748775e-6),
            np.complex(-1.2056784102984098e-8, -1.2210951690230782e-8),
        ],
        [
            0,
            0.5871298855433338,
            np.complex(-3.3729618710801137e-6, 2.4152360811650373e-6),
            np.complex(-0.5233926069798007, -0.13655786303346068),
            np.complex(-4.623380373113224e-6, 0.000010362902695259763),
            np.complex(-0.17909656013941788, -0.11916639160269833),
            np.complex(-3.399720873431807e-6, -7.125008373682292e-7),
            np.complex(0.04072119358712736, -0.3719310475303641),
            np.complex(-7.536125619789242e-6, 1.885248226837573e-6),
            np.complex(-0.11393851510585044, -0.3456924286310791),
            np.complex(-2.3915763815197452e-6, -4.2406689395594674e-7),
            np.complex(0.12820184730203607, 0.0935942533049232),
            np.complex(-1.5407293261691393e-6, -2.4673669087089514e-6),
            np.complex(-0.012272903377715643, -0.13317144020065683),
            np.complex(-1.1260776123106269e-6, -1.6865728072273087e-7),
            np.complex(-0.01013345155253134, -0.0240812705564227),
            np.complex(0.0, -1.4163391111474348e-7),
            np.complex(-0.003213070562510137, -0.012363639898516247),
            np.complex(-1.0619280312362908e-8, -1.2021213613319027e-7),
            np.complex(-0.002006756716685063, -0.0026636832583059812),
            np.complex(0.0, -4.509035934797572e-8),
            np.complex(-0.00048585160444833446, -0.0005014735884977489),
            np.complex(-1.2286988061034212e-8, -2.1199721851825594e-8),
            np.complex(-0.00010897007463988193, -0.00007018240288615613),
            np.complex(-1.2811279935244964e-8, -1.160553871672415e-8),
            np.complex(-0.00001785800494916693, -6.603027186486886e-6),
            -1.1639448324793031e-8,
            np.complex(-2.4097385882316104e-6, -3.5223103057306496e-7),
            -1.0792272866841885e-8,
            np.complex(-2.597671478115077e-7, 2.622928060603902e-8),
        ],
    ]
    all_num_codes = [states17, statesM, statesM2, statesP, statesP2]
    probs = all_num_codes[idx]
    return probs


def num(hilbert_size, probs=None, mu=0):
    """
    Generates the number states.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_

    Args:
        hilbert_size (int): Hilbert space dimension (cutoff). For the well defined
                            number states that we use here, the Hilbert space size 
                            should not be less than 32. If the probabilities are not
                            supplied then we will randomly select a set.
        probs (None, optional): Probabilitiy vector for the number state. If not supplied then a
                                random vector is selected from the five different sets from the function
                                `_get_num_prob`.
        mu (int, optional): Logical encoding (0/1)
                            default: 0
    
    Returns:
        :class:`qutip.Qobj`: Number state ket.
    
    """
    if (probs == None) and (hilbert_size < 32):
        err = "Specify a larger Hilbert size for default\n"
        err += "num state if probabilities are not specified\n"
        raise ValueError(err)

    state = fock(hilbert_size, 0) * 0

    if probs == None:
        probs = _get_num_prob(0)

    for n, p in enumerate(probs[mu]):
        state += p * fock(hilbert_size, n)
    ket = state.unit()
    return ket


def binomial(hilbert_size, S, N=None, mu=0):
    """
    Generates a binomial state.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
        hilbert_size (int): Hilbert space size (cutoff).
        S (int): An integer parameter specifying 
        N (None, optional): A non-negative integer which specifies the order to which we can
                            correct dephasing errors and is similar to the ´alpha´ parameter
                            for cat states.
        mu (int, optional): Logical encoding (0/1)
                            default: 0
    
    Returns:
        :class:`qutip.Qobj`: Binomial state ket.
    """
    if N == None:
        Nmax = int((hilbert_size) / (S + 1)) - 1
        try:
            N = np.random.randint(0, Nmax)
        except:
            N = Nmax

    c = 1 / sqrt(2 ** (N + 1))

    psi = 0 * fock(hilbert_size, 0)

    for m in range(N):
        psi += (
            c
            * ((-1) ** (mu * m))
            * np.sqrt(binom(N + 1, m))
            * fock(hilbert_size, (S + 1) * m)
        )
    ket = psi.unit()
    return ket


def gkp(hilbert_size, delta, mu=0, zrange=20):
    """Generates a GKP state. 

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
        hilbert_size (int): Hilbert space size (cutoff).
        delta (float): 
        mu (int, optional): Logical encoding (0/1)
                            default: 0
        zrange (int, optional): The number of lattice points to loop over to construct
                                the grid of states. This depends on the Hilbert space
                                size and the delta value.
                                default: 20
    
    Returns:
        :class:`qutip.Qobj`: GKP state.
    """
    gkp = 0 * coherent(hilbert_size, 0)

    c = np.sqrt(np.pi / 2)

    zrange = range(-20, 20)

    for n1 in zrange:
        for n2 in zrange:
            a = c * (2 * n1 + mu + 1j * n2)
            alpha = coherent(hilbert_size, a)
            gkp += (
                np.exp(-(delta ** 2) * np.abs(a) ** 2)
                * np.exp(-1j * c ** 2 * 2 * n1 * n2)
                * alpha
            )

    ket = gkp.unit()
    return ket