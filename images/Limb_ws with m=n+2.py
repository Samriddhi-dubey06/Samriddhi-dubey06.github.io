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
plt.scatter(WORKSPACE[0, :], WORKSPACE[1, :], color='purple', marker='.', s=10, label='Reachable Workspace')

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



# --- 2R Manipulator Workspace with Cable Redundancy (m=n+2) ---
# Constants for the 2R manipulator
L1 = 1  # Length of the first link in arbitrary units (e.g., meters)
L2 = 1  # Length of the second link in arbitrary units (e.g., meters)

# Resolution for generating the joint angle space
# The number of points to sample for each joint angle
reso1 = 300  # Resolution for the first joint angle (q1)
reso2 = 300  # Resolution for the second joint angle (q2)

# Joint angle limits in degrees
# These limits define the range of motion for each joint
theta_1_l = -180  # Lower bound for the first joint angle (q1)
theta_2_l = -180  # Lower bound for the second joint angle (q2)
theta_1_u = 180  # Upper bound for the first joint angle (q1)
theta_2_u = 180  # Upper bound for the second joint angle (q2)

# Create an array of joint angles for the first joint (q1) ranging from theta_1_l to theta_1_u
q1 = np.linspace(np.radians(theta_1_l), np.radians(theta_1_u), reso1, endpoint=True)

# Create an array of joint angles for the second joint (q2) ranging from theta_2_l to theta_2_u
q2 = np.linspace(np.radians(theta_2_l), np.radians(theta_2_u), reso2, endpoint=True)

# Variables definition
link_lengths = (L1, L2)  # Link lengths

# These points are defined in the Cartesian coordinate system with respect to the manipulator's base.
base_points_4_cables = [(1, 0), (2, 0), (-2, 0), (-1, 0)]  #  coordinates for four cable anchor points

# These are given as a fraction of the link lengths, where 0.8 means 80% of the way along each link.
attachment_points = [0.8, 0.8]  # Attachment points on both links (as a fraction of the link length)


def calculate_eta_matrix_4_cables(link_lengths, base_points, attachment_points, q1, q2):
    """
    Calculate the structure matrix for a 2R manipulator with 4 cables.

    Parameters:
    link_lengths (tuple): Lengths of the two links.
    base_points (list of tuples): Coordinates of the cable base points (4 cables).
    attachment_points (list): Points along the links where cables are attached, as a fraction of link length.
    q1, q2 (float): Joint angles in radians.

    Returns:
    tuple: The eta1 and eta2 vectors.
    """
    L1, L2 = link_lengths
    op1, ap2 = attachment_points  # Attachment points as fractions of link lengths

    # Calculate attachment point positions in Cartesian coordinates
    r1x = L1 * np.cos(q1) * op1
    r1y = L1 * np.sin(q1) * op1
    r2x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) * ap2
    r2y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) * ap2

    # Initialize the structure matrix A for 4 cables
    A = np.zeros((2, 4))  # 2 joints, 4 cables

    # Calculate partial derivatives of cable lengths w.r.t. q1 and q2 for 4 cables
    for i, (bx, by) in enumerate(base_points):
        if i == 0 or i == 3:  # First and fourth cable attached to the first link
            A[0, i] = ((bx - r1x) * (-L1 * np.sin(q1) * op1) + (by - r1y) * (L1 * np.cos(q1) * op1)) / np.hypot(
                r1x - bx, r1y - by)
            A[1, i] = 0
        else:  # Other cables attached to the second link
            A[0, i] = ((bx - r2x) * (-L1 * np.sin(q1) - L2 * np.sin(q1 + q2) * ap2) + (by - r2y) * (
                        L1 * np.cos(q1) + L2 * np.cos(q1 + q2) * ap2)) / np.hypot(r2x - bx, r2y - by)
            A[1, i] = ((bx - r2x) * (-L2 * np.sin(q1 + q2) * ap2) + (by - r2y) * (
                        L2 * np.cos(q1 + q2) * ap2)) / np.hypot(r2x - bx, r2y - by)

    # Create the column vectors for the A matrix
    A1 = A[:, 0].reshape(-1, 1)
    A2 = A[:, 1].reshape(-1, 1)
    A3 = A[:, 2].reshape(-1, 1)
    A4 = A[:, 3].reshape(-1, 1)

    # Calculate the eta1 and eta2 values
    eta1 = np.array([
        np.linalg.det(np.hstack((A3, A2))),
        np.linalg.det(np.hstack((A1, A3))),
        -np.linalg.det(np.hstack((A1, A2))),
        0
    ]).reshape(-1, 1)
    eta2 = np.array([
        np.linalg.det(np.hstack((A4, A2))),
        np.linalg.det(np.hstack((A1, A4))),
        0,
        -np.linalg.det(np.hstack((A1, A2)))
    ]).reshape(-1, 1)
    return eta1, eta2


plotspacepp, plotspacenn, plotspacepn, plotspacenp = [], [], [], []
for i in q1:
    for j in q2:
        eta1, eta2 = calculate_eta_matrix_4_cables(link_lengths, base_points_4_cables, attachment_points, i, j)

        # Check if all elements in eta1 and eta2 are positive or negative
        if np.all(eta1 >= 0) and np.all(eta2 >= 0):
            plotspacepp.append([i, j])
        elif np.all(eta1 <= 0) and np.all(eta2 <= 0):
            plotspacenn.append([i, j])
        elif np.all(eta1 >= 0) and np.all(eta2 <= 0):
            plotspacepn.append([i, j])
        elif np.all(eta1 <= 0) and np.all(eta2 >= 0):
            plotspacenp.append([i, j])

plotspaceP = np.degrees(np.array(plotspacepp))
plotspaceN = np.degrees(np.array(plotspacenn))
plotspaceNP = np.degrees(np.array(plotspacenp))
plotspacePN = np.degrees(np.array(plotspacepn))

# Initialize an empty list to hold the arrays that are not empty
non_empty_arrays = []

# Append non-empty arrays to the list
if len(plotspacepp) > 0:
    non_empty_arrays.append(plotspaceP)
if len(plotspacenn) > 0:
    non_empty_arrays.append(plotspaceN)
if len(plotspacenp) > 0:
    non_empty_arrays.append(plotspaceNP)
if len(plotspacepn) > 0:
    non_empty_arrays.append(plotspacePN)

# Convert lists to NumPy arrays and stack non-empty arrays vertically
if non_empty_arrays:  # Check if the list is not empty
    plotspace = np.vstack(non_empty_arrays)
else:
    # Handle the case where all arrays are empty
    plotspace = np.array([])  # Empty NumPy array

plt.figure(figsize=(10, 8))

# Initialize lists for storing all x and y coordinates
all_x, all_y = [], []


# Function to plot and collect coordinates if not empty
def plot_and_collect(data, color, label, marker='o', s=20, alpha=0.7):
    if data.size > 0:  # Check if the data array is not empty
        plt.scatter(data[:, 0], data[:, 1], color=color, marker=marker, s=s, label=label, alpha=alpha)
        return data[:, 0], data[:, 1]
    return [], []


# Plot each dataset and collect coordinates
x, y = plot_and_collect(plotspaceP, '#FF6B6B', 'Eta Both Positive')
all_x.extend(x);
all_y.extend(y)
x, y = plot_and_collect(plotspaceN, '#4ECDC4', 'Eta Both Negative')
all_x.extend(x);
all_y.extend(y)
x, y = plot_and_collect(plotspaceNP, 'b', 'Eta Mixed (Neg, Pos)')
all_x.extend(x);
all_y.extend(y)
x, y = plot_and_collect(plotspacePN, 'black', 'Eta Mixed (Pos, Neg)')
all_x.extend(x);
all_y.extend(y)

# Adding labels and title
plt.title('2R Manipulator Workspace [2 Redundancies Case (m=n+2)]', fontsize=16, fontweight='bold')
plt.xlabel('q1 (Degrees)', fontsize=14)
plt.ylabel('q2 (Degrees)', fontsize=14)

# Adding grid, legend, and setting the aspect ratio
plt.grid(True)
plt.legend(fontsize=12, loc='best')
plt.axis('equal')  # Ensuring equal scaling on both axes

# Dynamically setting the limits of the plot based on the collected data
if all_x and all_y:  # Check if lists are not empty
    plt.xlim([min(all_x), max(all_x)])
    plt.ylim([min(all_y), max(all_y)])
plt.show()





# --- Task Space of Cable-Driven 2R Manipulator ---
def forward_kinematics(q1, q2, l1=1, l2=1):
    """
    Calculate the Cartesian coordinates (x, y) of the end-effector (wrist position)
    for a 2-link planar manipulator (human hand) based on the provided joint angles.

    Parameters:
    q1 (float): The joint angle of the first link (thigh - Shoulder to elbow) in radians.
    q2 (float): The joint angle of the second link (shank - elbow to wrist) in radians.
    l1 (float, optional): The length of the first link (thigh). Defaults to 1 unit.
    l2 (float, optional): The length of the second link (shank). Defaults to 1 unit.

    Returns:
    np.array: A 2-element array containing the x and y coordinates of the wrist position.
    """

    # Calculate the x-coordinate using the sum of the projections of link lengths
    # on the x-axis based on their respective joint angles
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    # Calculate the y-coordinate using the sum of the projections of link lengths
    # on the y-axis based on their respective joint angles
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    # Return the Cartesian coordinates as a numpy array
    return np.array([x, y])


taskspace = []
if plotspace.size > 0:
    for i in plotspace:
        x = L1 * np.cos(np.radians(i[0])) + L2 * np.cos(np.radians(i[0]) + np.radians(i[1]))
        y = L1 * np.sin(np.radians(i[0])) + L2 * np.sin(np.radians(i[0]) + np.radians(i[1]))
        taskspace.append((x, y))

# Set up the plot with a specified figure size for better visibility
plt.figure(figsize=(10, 10))

# Scatter plot the task space data
# Unpack the taskspace list of tuples into x and y coordinates using zip and scatter plot them
if taskspace:
    plt.scatter(*zip(*taskspace), label='Modeled Cable-Driven (4 Cables) 2R Manipulator Workspace', alpha=0.6, s=50)

# Enhance the plot title with an informative and concise description
plt.title('Simulated Task Space of Cable-Driven 2R Manipulator)', 
          fontsize=18, fontweight='bold', color='darkred')

# Improve the axis labels to clearly indicate what the axes represent
plt.xlabel('X Coordinate (meters)', fontsize=16)  # Assuming the units are in meters
plt.ylabel('Y Coordinate (meters)', fontsize=16)  # Assuming the units are in meters

# Add a legend to identify the data points with an adjusted font size and a frame for readability
plt.legend(fontsize=14, frameon=True, shadow=True)

# Add grid lines to the plot for better readability of the coordinates
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot with the improved aesthetics
plt.show()







# --- Comparative Task Space Visualization ---

# Set up the plot with a specified figure size for better visibility
plt.figure(figsize=(10, 10))

# Scatter plot the task space data from the modelled cable-driven 2R manipulator
# Unpack the taskspace list of tuples into x and y coordinates using zip and scatter plot them
plt.scatter(*zip(*taskspace), label='(4 Cables) 2R Manipulator Workspace', alpha=0.7, s=40)

# Scatter plot the task space data from the human hand workspace
# Here, WORKSPACE[0, :] are the X coordinates and WORKSPACE[1, :] are the Y coordinates
plt.scatter(WORKSPACE[0, :], WORKSPACE[1, :], color='purple', marker='o', s=10, label='Human hand Workspace',
            alpha=0.7)

# Set the title of the plot with an informative and concise description
plt.title(' 2R Manipulator vs. Human Hand', 
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