import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visualization
sns.set(style='whitegrid')

# --- Defining the Kinematic Structure of the Human Arm ---

# Length of the upper arm (from shoulder to elbow) and forearm (from elbow to wrist)
l1 = 1  # Upper arm length (meters)
l2 = 1  # Forearm length (meters)

# Resolution settings for the generation of joint angle values
reso1 = 200  # Number of discrete samples for the shoulder joint angle
reso2 = 200  # Number of discrete samples for the elbow joint angle

# Joint angle limits (in degrees)
Theta_1_l = 0    # Minimum allowable shoulder joint angle
Theta_1_u = 180  # Maximum allowable shoulder joint angle
Theta_2_l = -45  # Minimum allowable elbow joint angle
theta_2_u = 145  # Maximum allowable elbow joint angle

# Function to perform forward kinematics and compute the Cartesian coordinates of the wrist
def forward_kinematics(q1, q2, l1=1, l2=1):
    """
    Computes the (x, y) position of the wrist given:
    - q1: Shoulder joint angle (in radians)
    - q2: Elbow joint angle (in radians)
    - l1: Length of the upper arm (default = 1 meter)
    - l2: Length of the forearm (default = 1 meter)
    
    Returns:
    - A NumPy array containing the x and y coordinates of the wrist.
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)  # X-coordinate of the wrist
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)  # Y-coordinate of the wrist
    return np.array([x, y])

# Generate a range of shoulder joint angles (converted to radians)
q11 = np.linspace(np.radians(Theta_1_l), np.radians(Theta_1_u), reso1, endpoint=True)

# Initialize an array to store all computed workspace coordinates
WORKSPACE = np.zeros((2, reso1 * reso2))

# Compute the reachable workspace by iterating over all possible joint angle combinations
index = 0  # Index counter for storing workspace coordinates
for shoulder_angle in q11:
    # Generate a range of elbow joint angles (converted to radians)
    q22 = np.linspace(np.radians(Theta_2_l), np.radians(theta_2_u), reso2, endpoint=True)
    for elbow_angle in q22:
        # Compute the (x, y) position of the wrist using forward kinematics
        WORKSPACE[:, index] = forward_kinematics(shoulder_angle, elbow_angle, l1, l2)
        index += 1

# --- Plotting the Workspace of the Human Arm ---

plt.figure(figsize=(8, 6))

# Plot the workspace points representing all reachable wrist positions
plt.scatter(WORKSPACE[0, :], WORKSPACE[1, :], color='orange', marker='.', s=10, label='Reachable Workspace')

# Mark the shoulder joint (origin of motion) in red
plt.plot(0, 0, 'ro', label='Shoulder Joint')

# Enhance the visualization with axis labels, title, and styling
plt.xlabel('X Position (meters)', fontsize=12)
plt.ylabel('Y Position (meters)', fontsize=12)
plt.title('Reachable Workspace of a 2-Link Arm in the Sagittal Plane', fontsize=14, fontweight='bold')

# Improve readability with a grid, legend, and equal aspect ratio for accurate proportions
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axis('equal')

# Display the workspace plot
plt.show()




# --- 2R Manipulator with Cable-Driven Actuation ---

# Link lengths (units: meters or arbitrary units)
L1 = 1  # Length of the first link
L2 = 1  # Length of the second link

# Resolution for joint angle sampling
reso1 = 200  # Number of samples for the first joint angle (q1)
reso2 = 200  # Number of samples for the second joint angle (q2)

# Joint angle limits (in degrees)
theta_1_l, theta_1_u = -180, 180  # Limits for the first joint
theta_2_l, theta_2_u = -180, 180  # Limits for the second joint

# Generate joint angle arrays (converted to radians)
q1 = np.linspace(np.radians(theta_1_l), np.radians(theta_1_u), reso1, endpoint=True)
q2 = np.linspace(np.radians(theta_2_l), np.radians(theta_2_u), reso2, endpoint=True)

# Define system parameters
link_lengths = (L1, L2)  # Tuple storing link lengths

# Cable anchor points in Cartesian coordinates relative to the base
base_points_3_cables = [(2, 0), (4, 0), (-4, 0)]

# Positions where cables attach to the links (expressed as fractions of link lengths)
attachment_points = [0.8, 0.8]  # 80% along both links

def calculate_eta_matrix_3_cables(link_lengths, base_points, attachment_points, q1, q2):
    """
    Computes the eta vector for a 2R manipulator actuated by 3 cables.
    The eta vector represents the null space of the structure matrix,
    helping analyze force distribution among the cables.
    
    Parameters:
    - link_lengths (tuple): Lengths of both links.
    - base_points (list of tuples): Cable anchor points in Cartesian coordinates.
    - attachment_points (list): Fractional positions along the links where cables attach.
    - q1, q2 (float): Joint angles in radians.
    
    Returns:
    - numpy.ndarray: Eta vector derived from the structure matrix.
    """
    L1, L2 = link_lengths
    op1, ap2 = attachment_points

    # Compute Cartesian coordinates of cable attachment points
    r1x, r1y = L1 * np.cos(q1) * op1, L1 * np.sin(q1) * op1
    r2x, r2y = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) * ap2, L1 * np.sin(q1) + L2 * np.sin(q1 + q2) * ap2

    # Initialize structure matrix (2x3 for 2 joints and 3 cables)
    A = np.zeros((2, 3))

    # Compute structure matrix elements
    for i, (bx, by) in enumerate(base_points):
        if i == 0:  # Cable attached to the first link
            A[0, i] = ((bx - r1x) * (-L1 * np.sin(q1) * op1) + (by - r1y) * (L1 * np.cos(q1) * op1)) / np.hypot(r1x - bx, r1y - by)
        else:  # Cables attached to the second link
            A[0, i] = ((bx - r2x) * (-L1 * np.sin(q1) - L2 * np.sin(q1 + q2) * ap2) + (by - r2y) * (L1 * np.cos(q1) + L2 * np.cos(q1 + q2) * ap2)) / np.hypot(r2x - bx, r2y - by)
            A[1, i] = ((bx - r2x) * (-L2 * np.sin(q1 + q2) * ap2) + (by - r2y) * (L2 * np.cos(q1 + q2) * ap2)) / np.hypot(r2x - bx, r2y - by)

    # Extract column vectors for determinant calculations
    A1, A2, A3 = A[:, 0].reshape(-1, 1), A[:, 1].reshape(-1, 1), A[:, 2].reshape(-1, 1)

    # Compute eta vector (determinants of submatrices formed by cable vectors)
    eta = np.array([
        np.linalg.det(np.concatenate([A3, A2], axis=1)),
        np.linalg.det(np.concatenate([A1, A3], axis=1)),
        -np.linalg.det(np.concatenate([A1, A2], axis=1))
    ]).reshape(-1, 1)

    return eta

# Store joint angle pairs where eta conditions are met
plotspacea, plotspaceb = [], []

# Iterate through all joint angle combinations
for i in q1:
    for j in q2:
        eta = calculate_eta_matrix_3_cables(link_lengths, base_points_3_cables, attachment_points, i, j)
        
        # Classify based on eta vector values
        if (eta[0] > 0 and eta[1] > 0 and eta[2] > 0):
            plotspacea.append([i, j])  # Positive eta condition
        elif (eta[0] < 0 and eta[1] < 0 and eta[2] < 0):
            plotspaceb.append([i, j])  # Negative eta condition

# Convert joint angle data from radians to degrees for visualization
plotspaceP1 = np.degrees(np.array(plotspacea))
plotspaceN1 = np.degrees(np.array(plotspaceb))
plotspace = np.vstack([plotspacea, plotspaceb])

# --- Visualization of the Feasible Workspace ---
plt.figure(figsize=(10, 8))

# Plot joint angle pairs where eta values are positive
plt.scatter(plotspaceP1[:, 0], plotspaceP1[:, 1], color='#FF6B6B', marker='o', s=10, label='Eta Positive', alpha=0.7)

# Plot joint angle pairs where eta values are negative
plt.scatter(plotspaceN1[:, 0], plotspaceN1[:, 1], color='#4ECDC4', marker='o', s=10, label='Eta Negative', alpha=0.7)

# Add labels and title
plt.title('2R Manipulator Workspace with Cable Redundancy', fontsize=16, fontweight='bold')
plt.xlabel('q1 (Degrees)', fontsize=14)
plt.ylabel('q2 (Degrees)', fontsize=14)

# Enable grid, legend, and aspect ratio adjustment
plt.grid(True)
plt.legend(fontsize=12)
plt.axis('equal')  # Ensuring uniform scaling for both axes

# Set plot limits based on data range
plt.xlim([min(plotspaceP1[:, 0].min(), plotspaceN1[:, 0].min()),
          max(plotspaceP1[:, 0].max(), plotspaceN1[:, 0].max())])
plt.ylim([min(plotspaceP1[:, 1].min(), plotspaceN1[:, 1].min()),
          max(plotspaceP1[:, 1].max(), plotspaceN1[:, 1].max())])

# Display the workspace plot
plt.show()









# --- Task Space of Cable-Driven 2R Manipulator ---
def forward_kinematics(q1, q2, l1=1, l2=1):
    """
    Computes the Cartesian coordinates (x, y) of the end-effector (wrist position)
    for a 2-link planar manipulator, given the joint angles.

    Parameters:
    q1 (float): Joint angle of the first link (shoulder to elbow) in radians.
    q2 (float): Joint angle of the second link (elbow to wrist) in radians.
    l1 (float, optional): Length of the first link (upper arm). Default is 1 unit.
    l2 (float, optional): Length of the second link (forearm). Default is 1 unit.

    Returns:
    np.array: A 2D array representing the (x, y) coordinates of the wrist.
    """

    # Compute the x-coordinate as the sum of x-projections of both links
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)

    # Compute the y-coordinate as the sum of y-projections of both links
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    # Return the calculated (x, y) position as a numpy array
    return np.array([x, y])


# Initialize an empty list to store the computed task space coordinates
taskspace = []

# Iterate through the predefined joint angle set (plotspace) to compute the workspace
for i in plotspace:
    x = L1 * np.cos(i[0]) + L2 * np.cos(i[0] + i[1])
    y = L1 * np.sin(i[0]) + L2 * np.sin(i[0] + i[1])
    taskspace.append((x, y))  # Store the computed (x, y) coordinates

# Set up the plot with a fixed size for better visualization
plt.figure(figsize=(10, 10))

# Generate a scatter plot of the computed task space points
# Unpack the list of tuples into separate x and y coordinate lists
plt.scatter(*zip(*taskspace), label='Simulated Task Space of Cable-Driven 2R Manipulator', 
            alpha=0.6, s=50)

# Define the plot title with a precise description of the workspace representation
plt.title('Task Space of 2R Manipulator (m=n+1) Redundancy)', 
          fontsize=18, fontweight='bold', color='darkred')

# Label the x-axis and y-axis to indicate spatial dimensions (assumed in meters)
plt.xlabel('X Position (meters)', fontsize=16)
plt.ylabel('Y Position (meters)', fontsize=16)

# Display a legend for clarity, with enhanced formatting for better readability
plt.legend(fontsize=14, frameon=True, shadow=True)

# Enable a dashed grid to improve coordinate readability
plt.grid(True, linestyle='--', alpha=0.7)

# Render the final plot with refined visual presentation
plt.show()






# --- Comparative Task Space Visualization ---

# Set up the plot with a specified figure size for better visibility
plt.figure(figsize=(10, 10))

# Scatter plot the task space data from the modelled cable-driven 2R manipulator
# Unpack the taskspace list of tuples into x and y coordinates using zip and scatter plot them
plt.scatter(*zip(*taskspace), label=' (3 Cables) 2R Manipulator Workspace', alpha=0.7, s=40)

# Scatter plot the task space data from the human hand workspace
# Here, WORKSPACE[0, :] are the X coordinates and WORKSPACE[1, :] are the Y coordinates
plt.scatter(WORKSPACE[0, :], WORKSPACE[1, :], color='orange', marker='o', s=10, label='Human hand Workspace',
            alpha=0.7)

# Set the title of the plot with an informative and concise description
plt.title('2R Manipulator vs. Human Hand', 
          fontsize=18, fontweight='bold', color='darkslategray')


# Set the x and y-axis labels with clear descriptions and larger font size for readability
plt.xlabel('X Coordinate (units)', fontsize=16)  # Specify the units if known
plt.ylabel('Y Coordinate (units)', fontsize=16)  # Specify the units if known

# Add a legend to the plot to differentiate between the two datasets plotted
plt.legend(fontsize=14, frameon=True, shadow=True)

# Add grid lines to the plot for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()
