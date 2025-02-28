import numpy as np
import matplotlib.pyplot as plt

transmissions = np.array([0.927633, 0.935765, 0.943876, 0.952080, 0.960390, 0.968736, 0.976918, 0.984558, 0.990949,0.994984, 0.996273, 0.994744, 0.98968, 0.980947, 0.969502, 0.956244, 0.941939, 0.927182, 0.912392, 0.897866, 0.883716])
charges = np.array([-1,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
plt.scatter(charges, transmissions)
plt.xlabel("Coulomb Charge eV")
plt.ylabel("Transmission")
plt.title("Coulomb Charge Vs transmission of Electron")
plt.show()