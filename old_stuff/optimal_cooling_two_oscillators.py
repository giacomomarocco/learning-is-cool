import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:27:18 2026

@author: giacomomarocco
"""

#!/usr/bin/env python3
"""
Cooling two non-degenerate oscillators with a common optimal feedback force.
Plots steady-state nbar (averaged over both oscillators) vs feedback gain g_fb.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import os
from gaussian_oscillator import GaussianOscillator
from optimal_control import FeedbackForces

dirname = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.linalg import solve_continuous_are

class OptimalFeedbackTwoOscillators:
    """
    Compute optimal LQR feedback for two non-degenerate oscillators
    subject to a common force.
    
    State vector: [xc1, pc1, xc2, pc2]
    """
    
    def __init__(self, omega1, omega2, q=1.0):
        """
        Parameters
        ----------
        omega1 : float
            Frequency of oscillator 1.
        omega2 : float
            Frequency of oscillator 2.
        q : float
            Control cost parameter (higher = more expensive control).
        """
        self.omega1 = omega1
        self.omega2 = omega2
        self.q = q
        
        # Build the 4x4 matrices
        self.A_full = self._build_A()
        self.B_full = self._build_B()
        self.P_full = self._build_P()
        self.Q_full = self._build_Q()
        
        # Solve CARE and get gain matrix
        self.U = self._solve_care()
        self.K = self._compute_gain()
    
    def _build_A(self):
        """Block-diagonal drift matrix: diag(A1, A2)."""
        A1 = np.array([[0, self.omega1],
                        [-self.omega1, 0]])
        A2 = np.array([[0, self.omega2],
                        [-self.omega2, 0]])
        A_full = np.zeros((4, 4))
        A_full[0:2, 0:2] = A1
        A_full[2:4, 2:4] = A2
        return A_full
    
    def _build_B(self):
        """Control matrix: (1/N) * (J_N ⊗ B), with N=2."""
        B_single = np.array([[0, 0],
                              [0, 1]])
        J2 = np.ones((2, 2))
        return 0.5 * np.kron(J2, B_single)  # 4x4
    
    def _build_P(self):
        """State cost: I_N ⊗ P, with P = omega * I_2 for each oscillator."""
        # Using P_a = omega_a * I_2 for each oscillator
        P_full = np.zeros((4, 4))
        P_full[0:2, 0:2] = self.omega1 * np.eye(2)
        P_full[2:4, 2:4] = self.omega2 * np.eye(2)
        return P_full
    
    def _build_Q(self):
        """Control cost: I_N ⊗ Q = I_N ⊗ (q * I_2)."""
        return self.q * np.eye(4)
    
    def _solve_care(self):
        """
        Solve the CARE:
          P + A^T U + U A - U B Q^{-1} B^T U = 0
        
        scipy's solve_continuous_are solves:
          A^T X + X A - X B R^{-1} B^T X + Q = 0
        with Q -> P, R -> Q_full here.
        """
        return solve_continuous_are(self.A_full, self.B_full, self.P_full, self.Q_full)
    
    def _compute_gain(self):
        """K = Q^{-1} B^T U."""
        Q_inv = np.linalg.inv(self.Q_full)
        return Q_inv @ self.B_full.T @ self.U
    
    def optimal_feedback(self):
        """
        Return a function f(xc1, pc1, xc2, pc2) -> scalar force
        that gives the optimal common force to apply.
        
        The control is u = -K @ x_c, and the actual force on each
        particle is the momentum component of (1/N)(J_N ⊗ B) @ u,
        which simplifies to a single scalar force.
        """
        K = self.K
        
        def feedback_fn(xc1, pc1, xc2, pc2):
            state = np.array([xc1, pc1, xc2, pc2])
            u = -K @ state
            # The common force on each particle is the p-component
            # of (1/N)(J_N ⊗ B) @ u. Since B only has a (1,1) entry,
            # the force on particle a is (1/N) * sum_b u_{p_b}.
            # u has components [u_x1, u_p1, u_x2, u_p2]; 
            # the force = 0.5*(u_p1 + u_p2) due to (1/N)*J structure.
            force = 0.5 * (u[1] + u[3])
            return force
        
        return feedback_fn
    
    def optimal_feedback_vec(self):
        """
        Return the effective gain vector g such that 
        force = -g @ [xc1, pc1, xc2, pc2].
        """
        K = self.K
        # force = -0.5 * (K[1,:] + K[3,:]) @ state
        g = 0.5 * (K[1, :] + K[3, :])
        return g


# --- Example usage ---
if __name__ == "__main__":
    omega1 = 1.0
    omega2 = 1.05  # slightly non-degenerate
    q = 1.0
    
    fb = OptimalFeedbackTwoOscillators(omega1, omega2, q)
    
    print("Gain matrix K:")
    print(fb.K)
    print()
    
    g = fb.optimal_feedback_vec()
    print(f"Effective gain vector: {g}")
    print(f"  force = -({g[0]:.4f} xc1 + {g[1]:.4f} pc1 + {g[2]:.4f} xc2 + {g[3]:.4f} pc2)")
    print()
    
    # Get the feedback function
    feedback_fn = fb.optimal_feedback()
    
    # Test it
    force = feedback_fn(1.0, 0.5, -0.3, 0.2)
    print(f"Force for state [1.0, 0.5, -0.3, 0.2]: {force:.4f}")