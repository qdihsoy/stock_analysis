import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0.1, 10, 100)

# U_risk_averse = np.sqrt(x) #リスク回避
# U_risk_neutral = x #リスク中立
# U_risk_seeking = x**1.5 #リスク追及

# plt.figure(figsize=(10, 6))
# plt.plot(x, U_risk_averse, label='リスク回避', linewidth=2)
# plt.plot(x, U_risk_neutral, label='リスク回避', linewidth=2)
# plt.plot(x, U_risk_seeking, label='リスク回避', linewidth=2)

# plt.ylim(0, 12)
# plt.title('投資家のリスク態度と効用関数', fontsize=14)
# plt.xlabel('収益x', fontsize=12)
# plt.ylabel('効用U(x)', fontsize=12)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#ギャンブル
x1, x2 = 2, 8
p = 0.5

#確実な収益
x_certain = 5

def U_risk_averse(x): return np.sqrt(x)
def U_risk_neutral(x): return x
def U_risk_seeking(x): return x**2

def EU(func):
  return p * func(x1) + (1-p) * func(x2)

#逆関数（効用➡金額）
def inv_U_risk_averse(u): return u**2
def inv_U_risk_neutral(u): return u
def inv_U_risk_seeking(u): return np.sqrt(u)

inv_funcs = {
  'リスク回避': inv_U_risk_averse,
  'リスク中立': inv_U_risk_neutral,
  'リスク追及': inv_U_risk_seeking
}

for name, func in {
  'リスク回避': U_risk_averse,
  'リスク中立': U_risk_neutral,
  'リスク追及': U_risk_seeking
}.items():
  eu_gamble = EU(func)
  u_certain = func(x_certain)
  decision = 'ギャンブル選好' if eu_gamble > u_certain else '確実な選択を好む'
  
  #確実性等価の計算
  inv_func = inv_funcs[name]
  ce = inv_func(eu_gamble)
  
  print(f'{name}:')
  print(f' 期待効用（ギャンブル）: {eu_gamble:.3f}')
  print(f' 効用（確実な5） : {u_certain:.3f}')
  print(f' 確実性等価: {ce:.3f}')
  print(f' リスクプレミアム: {eu_gamble - ce:.3f}')
  print(f' 結論: {decision}\n')