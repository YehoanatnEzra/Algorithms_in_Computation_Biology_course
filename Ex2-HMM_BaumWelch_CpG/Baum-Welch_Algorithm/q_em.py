import numpy as np
import os
import matplotlib.pyplot as plt

CONVERGENCE_THRESHOLD = 1e-4

# =====================================================
#                 FASTA SEQUENCE READER
# =====================================================
def read_fasta_sequences(path):
    """Reads FASTA sequences into a list of strings."""
    seqs = []
    with open(path) as f:
        current = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append("".join(current))
    return seqs


# =====================================================
#                   HMM_EM CLASS
# =====================================================
class HMM_EM:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations

        N = len(states)
        M = len(observations)

        # -------------------------------------------------------
        # INITIALIZATIONS (kept exactly as you requested)
        # -------------------------------------------------------
        # Initial state probabilities
        self.initial = np.array([0.5, 0.5])

        # Transition matrix
        self.transition = np.random.rand(N, N)
        self.transition = self.transition / self.transition.sum(axis=1, keepdims=True)

        # Emission matrix: 
        self.emission = np.random.rand(N, M)
        self.emission = self.emission / self.emission.sum(axis=1, keepdims=True)
        
        # Print initialized parameters (for debugging)
        print("Initial state distribution:\n", self.initial)
        print("Transition matrix:\n", self.transition)
        print("Emission matrix:\n", self.emission)

        # Observation → index lookup
        self.obs_index = {o: i for i, o in enumerate(observations)}

    # =====================================================
    #                   FORWARD ALGORITHM
    # =====================================================
    def forward(self, obs_seq):
        """
        compute forware probabilities a_t(i).

        Returns
        -------
        alpha : np.ndarray of shape (T, N)
        """
        T = len(obs_seq)
        N = len(self.states)
        alpha = np.zeros((T, N))

        # alpha[0, i] = pi_i * e_i(X_0)
        first_obs_idx = self.obs_index[obs_seq[0]]
        alpha[0, :] = self.initial * self.emission[:, first_obs_idx]

        # Recursion (t=1 to T-1)
        for t in range(1, T):
            obs_idx = self.obs_index[obs_seq[t]]
            alpha[t, :] = np.dot(alpha[t-1, :], self.transition) * self.emission[:, obs_idx]

        return alpha

    # =====================================================
    #                   BACKWARD ALGORITHM
    # =====================================================
    def backward(self, obs_seq):
        """
        Compute backward probabilities β_t(i).

        Returns
        -------
        beta : np.ndarray of shape (T, N)
        """
        T = len(obs_seq)
        N = len(self.states)
        beta = np.zeros((T, N))

        # Initialization (t=T-1), beta[T-1, i] = 1 for all i
        beta[T-1, :] = 1.0

        # Recursion (t=T-2 down to 0)
        for t in range(T - 2, -1, -1):
            next_obs_idx = self.obs_index[obs_seq[t+1]]
            # beta[t, i] = sum_j ( transition[i,j] * emission[j, next_obs] * beta[t+1, j] )
            beta[t, :] = np.dot(self.transition, (self.emission[:, next_obs_idx] * beta[t+1, :]))

        return beta

    # =====================================================
    #                   BAUM–WELCH / EM
    # =====================================================
    def baum_welch(self, seqs, max_iter=150):
        """
        Train the HMM using Baum–Welch EM.

        Returns
        -------
        transition : np.ndarray
        emission : np.ndarray
        """
        N = len(self.states)
        M = len(self.observations)
        log_likelihood_history = []

        for iteration in range(max_iter):
            # Initialize Accumulators for M-Step 
            total_gamma_init = np.zeros(N)
            
            # For Tau (Transition): Numerator (Xi) and Denominator (Gamma excluding last T)
            total_xi = np.zeros((N, N))
            total_gamma_transition_denom = np.zeros(N) 
            
            # For E (Emission): Numerator (Gamma per obs) and Denominator (Gamma total)
            total_gamma_obs = np.zeros((N, M))
            total_gamma_count = np.zeros(N)

            current_log_likelihood = 0

            # Iterate over all sequences ---
            for seq in seqs:
                T = len(seq)
                obs_indices = [self.obs_index[char] for char in seq]

                # Calculate Alpha and Beta
                alpha = self.forward(seq)
                beta = self.backward(seq)
                # Calculate Sequence Probability P(X|theta)
                seq_prob = np.sum(alpha[T-1, :])
                # Avoid division by zero if prob is extremely small
                if seq_prob == 0:
                    continue
                
                current_log_likelihood += np.log(seq_prob)

                # Calculate Gamma (State Probabilities)
                gamma = (alpha * beta) / seq_prob

                # Calculate Xi (Transition Probabilities) and Accumulate
                xi_acc = np.zeros((N, N))
                
                for t in range(T - 1):
                    curr_obs_idx = obs_indices[t+1]
                    # xi[t, i, j] calculation:
                    for i in range(N):
                        # Vectorized over j
                        xi_curr = alpha[t, i] * self.transition[i, :] * self.emission[:, curr_obs_idx] * beta[t+1, :]
                        xi_acc[i, :] += xi_curr / seq_prob
                
                # For Initial Probabilities (pi)
                total_gamma_init += gamma[0, :]

                # For Transition Probabilities (tau)
                total_xi += xi_acc
                total_gamma_transition_denom += np.sum(gamma[:-1, :], axis=0) # Sum gamma from 0 to T-2

                # For Emission Probabilities (e)
                total_gamma_count += np.sum(gamma, axis=0) # Total expected count for each state
                for t in range(T):
                    obs_idx = obs_indices[t]
                    total_gamma_obs[:, obs_idx] += gamma[t, :]
            
            # update parameters - M-Step
            self.initial = total_gamma_init / np.sum(total_gamma_init) # update initial
            self.transition = total_xi / total_gamma_transition_denom[:, np.newaxis] # update transition
            self.emission = total_gamma_obs / total_gamma_count[:, np.newaxis] # update emission

            # Store log-likelihood for convergence check
            log_likelihood_history.append(current_log_likelihood)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-Likelihood = {current_log_likelihood:.4f}")
            
            # Check for convergence (if change is very smaller than a threshold (1e-4))
            if iteration > 0 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < CONVERGENCE_THRESHOLD:
                print(f"Converged at iteration {iteration}")
                break

        return self.transition, self.emission, log_likelihood_history


# =============================================================
#                   MAIN PROGRAM
# =============================================================
if __name__ == "__main__":
    file_path = os.path.join("data", "q_coins.seq.fa")
    sequences = read_fasta_sequences(file_path)
    hmm = HMM_EM(states=["Fair", "Loaded"],observations=["H", "T"])

    try:
        # Run Baum-Welch 
        transition, emission, history = hmm.baum_welch(sequences, max_iter=150)

        print("\n" + "="*40)
        print(" FINAL RESULTS:")
        
        print("\nLearned Transition Matrix:")
        print("        To Fair   To Loaded")
        print(f"From F  {transition[0,0]:.4f}    {transition[0,1]:.4f}")
        print(f"From L {transition[1,0]:.4f}    {transition[1,1]:.4f}")

        print("\nLearned Emission Matrix:")
        print("        Emit H    Emit T")
        print(f"State F {emission[0,0]:.4f}    {emission[0,1]:.4f}")
        print(f"State L {emission[1,0]:.4f}    {emission[1,1]:.4f}")
        
        # Plot log-likelihood history        
        plt.figure(figsize=(10, 6))
        plt.plot(history, marker='o', linestyle='-', color='b')
        plt.title('Log-Likelihood Convergence over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.show()

    except NotImplementedError as e:
        print("Missing implementation:", e)
    except Exception as e:
        print("An error occurred:", e)