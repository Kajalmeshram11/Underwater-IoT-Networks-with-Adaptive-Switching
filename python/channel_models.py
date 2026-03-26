"""
=============================================================================
HYBRID OPTICAL-ACOUSTIC UNDERWATER IoT COMMUNICATION
Channel Models Module
=============================================================================
Paper: "Hybrid Optical–Acoustic Communication Protocol for Energy-Efficient
        High-Data-Rate Underwater IoT Networks with Adaptive Switching"
=============================================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# WATER TYPE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WATER_TYPES = {
    'pure_sea':    {'c': 0.005, 'b': 0.002, 'label': 'Pure Sea Water'},
    'clear_ocean': {'c': 0.114, 'b': 0.037, 'label': 'Clear Ocean'},
    'coastal':     {'c': 0.179, 'b': 0.219, 'label': 'Coastal Water'},
    'turbid':      {'c': 2.190, 'b': 1.824, 'label': 'Turbid/Harbour'},
}

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL CHANNEL MODEL  (Beer-Lambert Law)
# ─────────────────────────────────────────────────────────────────────────────
class OpticalChannel:
    """
    Models underwater optical wireless communication (UOWC).
    Uses Beer-Lambert attenuation law.

    Attenuation:  h(d) = exp(-c * d)
    where c = absorption + scattering coefficient (m⁻¹)
    """

    # Typical blue-green wavelength (optimal for UOWC)
    WAVELENGTH_NM   = 532        # nm
    MAX_RANGE_M     = 100        # meters (hard limit for optical)
    DATA_RATE_BPS   = 10e9       # 10 Gbps (max)
    TX_POWER_W      = 1.0        # Watts transmit power
    RX_SENSITIVITY  = -40        # dBm receiver sensitivity
    LATENCY_BASE_US = 0.5        # microseconds base latency

    def __init__(self, water_type='clear_ocean'):
        self.water = WATER_TYPES[water_type]
        self.c = self.water['c']   # total attenuation coefficient
        self.b = self.water['b']   # scattering coefficient

    def attenuation_db(self, distance_m):
        """Returns path loss in dB using Beer-Lambert law."""
        if distance_m <= 0:
            return 0.0
        # Beer-Lambert: Loss(dB) = 4.343 * c * d
        return 4.343 * self.c * distance_m

    def received_power_dbm(self, distance_m, tx_power_dbm=0):
        """Received power after propagation loss."""
        loss = self.attenuation_db(distance_m)
        rx = tx_power_dbm - loss
        return rx

    def snr_db(self, distance_m):
        """Estimate SNR at receiver."""
        noise_floor = -80  # dBm
        rx_power = self.received_power_dbm(distance_m)
        return rx_power - noise_floor

    def ber(self, distance_m):
        """
        Bit Error Rate using Q-function approximation.
        BER = 0.5 * erfc(sqrt(SNR/2))
        """
        from scipy.special import erfc
        snr_linear = 10 ** (self.snr_db(distance_m) / 10)
        snr_linear = max(snr_linear, 1e-10)
        return 0.5 * erfc(np.sqrt(snr_linear / 2))

    def achievable_data_rate(self, distance_m):
        """
        Shannon capacity-based achievable data rate (bps).
        Degrades with distance due to SNR loss.
        """
        if distance_m > self.MAX_RANGE_M:
            return 0.0
        snr_linear = 10 ** (self.snr_db(distance_m) / 10)
        bandwidth = 10e9  # 10 GHz optical bandwidth
        rate = bandwidth * np.log2(1 + max(snr_linear, 0))
        return min(rate, self.DATA_RATE_BPS)

    def energy_per_bit(self, distance_m):
        """Energy consumption per bit (Joules/bit)."""
        if distance_m > self.MAX_RANGE_M:
            return float('inf')
        rate = self.achievable_data_rate(distance_m)
        if rate <= 0:
            return float('inf')
        # E = P * T = P / R
        return self.TX_POWER_W / rate

    def latency(self, distance_m):
        """End-to-end latency in milliseconds."""
        # Speed of light in water ≈ 2.25×10⁸ m/s
        propagation_ms = (distance_m / 2.25e8) * 1000
        return self.LATENCY_BASE_US / 1000 + propagation_ms

    def is_feasible(self, distance_m, turbidity_factor=1.0):
        """Check if optical link is feasible."""
        effective_range = self.MAX_RANGE_M / turbidity_factor
        return distance_m <= effective_range

    def link_quality(self, distance_m):
        """
        Returns normalized link quality score 0-1.
        1 = excellent, 0 = infeasible
        """
        if not self.is_feasible(distance_m):
            return 0.0
        snr = self.snr_db(distance_m)
        # Map SNR to quality score (0 dB → 0, 30 dB → 1)
        quality = np.clip((snr + 10) / 40, 0, 1)
        return quality


# ─────────────────────────────────────────────────────────────────────────────
# ACOUSTIC CHANNEL MODEL  (Thorp's Formula)
# ─────────────────────────────────────────────────────────────────────────────
class AcousticChannel:
    """
    Models underwater acoustic communication (UAC).
    Uses Thorp's attenuation formula.

    Thorp absorption (dB/km):
      α(f) = 0.11*f²/(1+f²) + 44*f²/(4100+f²) + 2.75×10⁻⁴*f² + 0.003
    where f is frequency in kHz.

    Total path loss (dB):
      PL = k*10*log(d) + d*α(f)/1000
    """

    FREQ_KHZ       = 25.0        # kHz (typical OFDM underwater modem)
    MAX_RANGE_M    = 5000        # 5 km
    DATA_RATE_BPS  = 100e3       # 100 Kbps (max)
    TX_POWER_W     = 50.0        # Watts (higher than optical)
    SPREADING_K    = 1.5         # spherical spreading factor
    NOISE_LEVEL_DB = 50          # dB re μPa ambient noise

    def __init__(self, frequency_khz=25.0):
        self.freq = frequency_khz

    def thorp_absorption_db_per_km(self):
        """
        Thorp's empirical absorption coefficient (dB/km).
        Valid for f > 0.4 kHz.
        """
        f = self.freq
        a = (0.11 * f**2 / (1 + f**2) +
             44.0 * f**2 / (4100 + f**2) +
             2.75e-4 * f**2 +
             0.003)
        return a

    def path_loss_db(self, distance_m):
        """
        Total acoustic path loss using Thorp + spreading.
        PL(dB) = k*10*log10(d) + alpha*d/1000
        """
        if distance_m <= 0:
            return 0.0
        d_km = distance_m / 1000.0
        spreading_loss = self.SPREADING_K * 10 * np.log10(distance_m)
        absorption_loss = self.thorp_absorption_db_per_km() * d_km
        return spreading_loss + absorption_loss

    def snr_db(self, distance_m):
        """SNR at receiver (dB)."""
        tx_power_db = 10 * np.log10(self.TX_POWER_W * 1e6)  # dB re μW
        pl = self.path_loss_db(distance_m)
        return tx_power_db - pl - self.NOISE_LEVEL_DB

    def ber(self, distance_m):
        """BER estimate for BPSK modulation."""
        from scipy.special import erfc
        snr_linear = 10 ** (self.snr_db(distance_m) / 10)
        snr_linear = max(snr_linear, 1e-10)
        return 0.5 * erfc(np.sqrt(snr_linear))

    def achievable_data_rate(self, distance_m):
        """Shannon capacity (bps) for acoustic channel."""
        snr_linear = 10 ** (self.snr_db(distance_m) / 10)
        bandwidth = 30e3  # 30 kHz typical acoustic bandwidth
        rate = bandwidth * np.log2(1 + max(snr_linear, 0))
        return min(rate, self.DATA_RATE_BPS)

    def energy_per_bit(self, distance_m):
        """Energy consumption per bit (Joules/bit)."""
        rate = self.achievable_data_rate(distance_m)
        if rate <= 0:
            return float('inf')
        return self.TX_POWER_W / rate

    def latency(self, distance_m):
        """End-to-end latency in milliseconds."""
        # Speed of sound in seawater ≈ 1500 m/s
        propagation_ms = (distance_m / 1500.0) * 1000
        processing_ms  = 5.0   # modem processing delay
        return propagation_ms + processing_ms

    def link_quality(self, distance_m):
        """Returns normalized link quality score 0-1."""
        if distance_m > self.MAX_RANGE_M:
            return 0.0
        snr = self.snr_db(distance_m)
        quality = np.clip((snr + 5) / 35, 0, 1)
        return quality

    def is_feasible(self, distance_m):
        return distance_m <= self.MAX_RANGE_M


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL STATE (used by Q-Learning agent)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelState:
    """
    Encapsulates the full observable state of the underwater channel.
    Used as input to the RL decision agent.
    """
    def __init__(self, distance_m, residual_energy_j,
                 data_size_kb, turbidity_factor,
                 water_type='clear_ocean'):
        self.distance        = distance_m
        self.energy          = residual_energy_j
        self.data_size       = data_size_kb
        self.turbidity       = turbidity_factor
        self.water_type      = water_type
        self.optical_ch      = OpticalChannel(water_type)
        self.acoustic_ch     = AcousticChannel()

    def optical_feasible(self):
        return self.optical_ch.is_feasible(self.distance, self.turbidity)

    def acoustic_feasible(self):
        return self.acoustic_ch.is_feasible(self.distance)

    def to_feature_vector(self):
        """Normalized feature vector for ML input."""
        opt_quality = self.optical_ch.link_quality(self.distance)
        aco_quality = self.acoustic_ch.link_quality(self.distance)
        return np.array([
            self.distance / 5000.0,               # normalized distance
            self.energy / 100.0,                  # normalized energy (J)
            self.data_size / 1000.0,              # normalized data size
            self.turbidity / 5.0,                 # normalized turbidity
            float(self.optical_feasible()),        # optical feasibility
            float(self.acoustic_feasible()),       # acoustic feasibility
            opt_quality,                           # optical link quality
            aco_quality,                           # acoustic link quality
        ])

    def __repr__(self):
        return (f"ChannelState(d={self.distance:.0f}m, "
                f"E={self.energy:.1f}J, "
                f"data={self.data_size:.0f}KB, "
                f"turbidity={self.turbidity:.2f})")
