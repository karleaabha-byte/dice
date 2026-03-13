# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Stochastic Dice Simulator", layout="wide")
st.title("🎲 Advanced Stochastic Dice Simulator with Stochastic Processes & CLT")

# --- User Inputs ---
num_rolls = st.slider("Number of dice rolls:", min_value=50, max_value=10000, value=1000, step=50)
st.write(f"Simulating {num_rolls} rolls of a 6-sided die...")

bias_option = st.selectbox("Choose die type:", ["Fair Die", "Biased Die"])
memory_option = st.checkbox("Enable Markov chain memory (stochastic dependency)?", value=True)

# --- Define transition matrix if memory is enabled ---
if memory_option:
    st.write("Using a Markov chain transition matrix for stochastic dependency.")
    # Random example bias matrix: each row sums to 1
    transition_matrix = np.array([
        [0.1, 0.2, 0.2, 0.2, 0.2, 0.1],
        [0.1, 0.1, 0.3, 0.2, 0.2, 0.1],
        [0.15, 0.15, 0.1, 0.25, 0.2, 0.15],
        [0.1, 0.2, 0.2, 0.1, 0.3, 0.1],
        [0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
        [0.15, 0.15, 0.2, 0.2, 0.1, 0.2]
    ])
else:
    transition_matrix = None

# --- Generate dice rolls ---
rolls = []
current_state = np.random.randint(0,6)
rolls.append(current_state + 1)

for _ in range(1, num_rolls):
    if transition_matrix is not None:
        # Markov chain: next state depends on current
        current_state = np.random.choice(6, p=transition_matrix[current_state])
    else:
        # Independent roll
        if bias_option == "Fair Die":
            current_state = np.random.randint(0,6)
        else:
            # Example biased probabilities for biased die
            current_state = np.random.choice(6, p=[0.05,0.10,0.15,0.20,0.25,0.25])
    rolls.append(current_state + 1)

df = pd.DataFrame(rolls, columns=["Roll"])

# --- Frequency Analysis ---
st.subheader("Roll Frequencies")
counts = df['Roll'].value_counts().sort_index()
st.bar_chart(counts)

st.subheader("Estimated Probabilities")
probabilities = counts / num_rolls
st.bar_chart(probabilities)

# --- Cumulative Probability Convergence ---
st.subheader("Cumulative Probability Convergence")
cum_df = df.copy()
for face in range(1,7):
    cum_df[f'Face {face}'] = (df['Roll']==face).cumsum() / (np.arange(num_rolls)+1)
st.line_chart(cum_df[[f'Face {i}' for i in range(1,7)]])

# --- Random Walk of Dice Sums ---
st.subheader("Random Walk of Cumulative Sum of Rolls")
cum_sum = df['Roll'].cumsum()
st.line_chart(cum_sum)

# --- CLT Analysis ---
st.subheader("Central Limit Theorem Analysis (Sum of Rolls)")

# Rolling windows for sum
window_size = st.slider("CLT window size (number of rolls to sum):", min_value=10, max_value=500, value=50, step=10)

sum_windows = [df['Roll'][i:i+window_size].sum() for i in range(num_rolls - window_size + 1)]
mean_sum = np.mean(sum_windows)
std_sum = np.std(sum_windows)

st.write(f"Mean of window sums: {mean_sum:.2f}")
st.write(f"Standard deviation of window sums: {std_sum:.2f}")

fig, ax = plt.subplots()
ax.hist(sum_windows, bins=20, density=True, alpha=0.7, color='skyblue')
ax.set_title(f"Histogram of sums over rolling window size {window_size}")
ax.set_xlabel("Sum of Rolls")
ax.set_ylabel("Probability Density")
st.pyplot(fig)

st.write("""
🔹 **Observation**: According to the Central Limit Theorem, the distribution of the sum of a sufficiently large number of die rolls tends to a **normal distribution**, even if individual rolls are not normal.
""")
