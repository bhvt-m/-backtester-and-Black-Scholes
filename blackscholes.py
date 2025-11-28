import numpy as np
import math 
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
import matplotlib.pyplot as plt



# variables 

S = 1009.05
K = 1000.0
t = 30
T = t/365 

r = 0.03
sigma = 0.1541

flag = 'c'  #call

# math
def norm_cdf(x: float) -> float:
    
     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
 
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    
    if T <= 0:
        # At expiry, it's just intrinsic value
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
   
    call = bs_call_price(S, K, T, r, sigma)
    put = call - S + K * math.exp(-r * T)
    return put

call_price_manual = bs_call_price(S, K, T, r, sigma)
put_price_manual = bs_put_price(S, K, T, r, sigma)

print("\nManual Black-Scholes:")
print(f"Call price  (C) = {call_price_manual:.4f}")
print(f"Put price   (P) = {put_price_manual:.4f}")


#model role 


call_price_lib = black_scholes(flag, S, K, T, r, sigma)
put_price_lib  = black_scholes('p', S, K, T, r, sigma)

print("\npy_vollib Black-Scholes:")
print(f"Call price  (C) = {call_price_lib:.4f}")
print(f"Put price   (P) = {put_price_lib:.4f}")

call_delta = delta('c', S, K, T, r, sigma)
call_gamma = gamma('c', S, K, T, r, sigma)
call_vega  = vega('c', S, K, T, r, sigma)
call_theta = theta('c', S, K, T, r, sigma)
call_rho   = rho('c', S, K, T, r, sigma)

print("\nCall Greeks (py_vollib):")
print(f"Delta = {call_delta:.4f}")
print(f"Gamma = {call_gamma:.6f}")
print(f"Vega  = {call_vega:.4f}")
print(f"Theta = {call_theta:.4f}")
print(f"Rho   = {call_rho:.4f}")


# ---------- 5. P&L for a short call at expiry ----------

premium = call_price_lib  # what we receive today for selling 1 call

# Simulate a range of future HDFC prices at expiry
S_T = np.linspace(600, 1400, 200)  # from 600 to 1400

# Payoff of LONG call at expiry: max(S_T - K, 0)
long_call_payoff = np.maximum(S_T - K, 0.0)

# Payoff of SHORT call at expiry: you sold it, so negative of long payoff
short_call_payoff = -long_call_payoff

# Profit including premium received today (ignoring discounting for simplicity)
short_call_profit = premium + short_call_payoff

plt.figure()
plt.axhline(0, linestyle='--')
plt.axvline(K, linestyle='--', label='Strike K')

plt.plot(S_T, short_call_profit, label='Short Call P&L at Expiry')

plt.title("Short Call P&L at Expiry (HDFC Bank, 30 days)")
plt.xlabel("HDFC Price at Expiry (S_T)")
plt.ylabel("Profit / Loss")
plt.legend()
plt.grid(True)

plt.show()

print(f"\nIf you SHORT this call, you receive premium ≈ {premium:.2f} today.")
