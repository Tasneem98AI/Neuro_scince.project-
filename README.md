# Neuro_scince.project-
Ass  Lec 1
# In[1]:


import numpy as np

def tanh(x):
    return np.tanh(x)
#input
i1, i2 = 0.05, 0.10

#bais
b1, b2 = 0.5, 0.7

# Random weight 
np.random.seed(42)  # For reproducibility
weights = {
    'w1': np.random.uniform(-0.5, 0.5),
    'w2': np.random.uniform(-0.5, 0.5),
    'w3': np.random.uniform(-0.5, 0.5),
    'w4': np.random.uniform(-0.5, 0.5),
    'w5': np.random.uniform(-0.5, 0.5),
    'w6': np.random.uniform(-0.5, 0.5),
    'w7': np.random.uniform(-0.5, 0.5),
    'w8': np.random.uniform(-0.5, 0.5)
}



# In[3]:


# hidden layer 
h1_input = i1 * weights['w1'] + i2 * weights['w3'] + b1
h2_input = i1 * weights['w2'] + i2 * weights['w4'] + b1

h1_output = tanh(h1_input)
h2_output = tanh(h2_input)


# In[4]:


# output layer 
o1_input = h1_output * weights['w5'] + h2_output * weights['w7'] + b2
o2_input = h1_output * weights['w6'] + h2_output * weights['w8'] + b2

o1_output = tanh(o1_input)
o2_output = tanh(o2_input)


# In[5]:


print("Hidden Layer Outputs:")
print(f"h1: {h1_output}")
print(f"h2: {h2_output}")

print("\nOutput Layer Outputs:")
print(f"o1: {o1_output}")
print(f"o2: {o2_output}")

