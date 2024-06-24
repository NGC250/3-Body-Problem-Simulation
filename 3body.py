from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1 #6.67e-11 in SI units

class Dynamics:
    
    def __init__(self, masses, positions, initial_velocity, tick):
        
        self.masses = masses
        self.m1 = masses[0]
        self.m2 = masses[1]
        self.m3 = masses[2]
        
        self.initial_position = np.copy(positions)
        self.positions = positions
        self.r1 = positions[0]
        self.r2 = positions[1]
        self.r3 = positions[2]
        
        self.initial_velocity = np.copy(initial_velocity)
        self.velocities = initial_velocity
        self.tick = tick

    def Force(self, mass1, mass2, mass3, pos1, pos2, pos3):
        
        def Norm(x):
            norm = np.linalg.norm(x)
            return x / norm**3 if norm != 0 else np.zeros_like(x)
        
        Fg1 = G * mass1 * mass2 * Norm(pos2 - pos1)
        Fg2 = G * mass1 * mass3 * Norm(pos3 - pos1)
        Fg = Fg1 + Fg2
        
        return Fg
    
    def Netforce(self):
        
        F1 = self.Force(self.m1,self.m2,self.m3,self.r1,self.r2,self.r3)
        F2 = self.Force(self.m2,self.m3,self.m1,self.r2,self.r3,self.r1)
        F3 = self.Force(self.m3,self.m1,self.m2,self.r3,self.r1,self.r2)
        
        return np.vstack((F1,F2,F3))

    def Velocity(self):
        
        self.velocities = self.velocities + self.tick * self.Netforce()/self.masses
        return self.velocities
        
    def Evolution(self):
        
        forces = self.Netforce()
        self.positions = self.positions + self.tick * self.velocities + 0.5 * self.tick**2 * self.Netforce()/self.masses
        updated_forces = self.Netforce()
        
        self.velocities = self.velocities + 0.5 * self.tick * (forces + updated_forces)/self.masses

        self.r1, self.r2, self.r3 = self.positions[0], self.positions[1], self.positions[2]
        
        center_of_mass = (self.m1 * self.r1 + self.m2 * self.r2 + self.m3 * self.r3)/(np.sum(self.masses))
                    
        return self.positions, center_of_mass

class Animate2D:
    
    def __init__(self, dynamics):
        
        self.dynamics = dynamics
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Three-Body Problem Simulation')
        self.ax.grid(True)

        self.line1, = self.ax.plot([], [], 'r-', label='Body 1 Path')
        self.line2, = self.ax.plot([], [], 'g-', label='Body 2 Path')
        self.line3, = self.ax.plot([], [], 'b-', label='Body 3 Path')

        self.body1 = self.ax.plot([], [], 'ro', markersize=15)[0]
        self.body2 = self.ax.plot([], [], 'go', markersize=15)[0]
        self.body3 = self.ax.plot([], [], 'bo', markersize=15)[0]
        
        self.center_of_mass_marker = self.ax.plot([], [], 'ko', markersize=5, label='Center of mass')[0]

        self.pos1_hist,self.pos2_hist,self.pos3_hist = [],[],[]
    
    def init(self):
        
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.body1.set_data([], [])
        self.body2.set_data([], [])
        self.body3.set_data([], [])
        self.center_of_mass_marker.set_data([], [])

        return self.line1, self.line2, self.line3, self.body1, self.body2, self.body3, self.center_of_mass_marker

    def update(self, frame):
        
        positions, center_of_mass = self.dynamics.Evolution()
        self.pos1_hist.append(positions[0])
        self.pos2_hist.append(positions[1])
        self.pos3_hist.append(positions[2])
        
        pos1_hist_arr = np.array(self.pos1_hist)
        pos2_hist_arr = np.array(self.pos2_hist)
        pos3_hist_arr = np.array(self.pos3_hist)
        
        self.line1.set_data(pos1_hist_arr[:, 0], pos1_hist_arr[:, 1])
        self.line2.set_data(pos2_hist_arr[:, 0], pos2_hist_arr[:, 1])
        self.line3.set_data(pos3_hist_arr[:, 0], pos3_hist_arr[:, 1])
        self.body1.set_data([positions[0, 0]], [positions[0, 1]])
        self.body2.set_data([positions[1, 0]], [positions[1, 1]])
        self.body3.set_data([positions[2, 0]], [positions[2, 1]])
        
        self.center_of_mass_marker.set_data([center_of_mass[0]], [center_of_mass[1]])

        return self.line1, self.line2, self.line3, self.body1, self.body2, self.body3, self.center_of_mass_marker

    def start_animation(self):
        
        self.ani = FuncAnimation(self.fig, self.update, frames=iter(int, 1), cache_frame_data=False, init_func=self.init, blit=True, repeat=False)
        plt.legend()
        plt.show()

class Animate3D:
    
    def __init__(self, dynamics):
        
        self.dynamics = dynamics
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')  # Create a 3D subplot
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_zlim(-20, 20)  # Set limits for x, y, and z axes
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Three-Body Problem Simulation')
        self.ax.grid(True)

        self.line1, = self.ax.plot([], [], [], 'r-', label='Body 1 Path')
        self.line2, = self.ax.plot([], [], [], 'g-', label='Body 2 Path')
        self.line3, = self.ax.plot([], [], [], 'b-', label='Body 3 Path')

        self.body1, = self.ax.plot([], [], [], 'ro', markersize=15)
        self.body2, = self.ax.plot([], [], [], 'go', markersize=15)
        self.body3, = self.ax.plot([], [], [], 'bo', markersize=15)
        
        self.center_of_mass_marker = self.ax.plot([], [], 'ko', markersize=5, label='Center of mass')[0]

        self.pos1_hist, self.pos2_hist, self.pos3_hist = [], [], []
    
    def init(self):
        
        self.line1.set_data([], [])
        self.line1.set_3d_properties([])
        self.line2.set_data([], [])
        self.line2.set_3d_properties([])
        self.line3.set_data([], [])
        self.line3.set_3d_properties([])
        
        self.body1.set_data([], [])
        self.body1.set_3d_properties([])
        self.body2.set_data([], [])
        self.body2.set_3d_properties([])
        self.body3.set_data([], [])
        self.body3.set_3d_properties([])
        self.center_of_mass_marker.set_data([], [])
        self.center_of_mass_marker.set_3d_properties([])

        return self.line1, self.line2, self.line3, self.body1, self.body2, self.body3, self.center_of_mass_marker

    def update(self, frame):
        
        positions, center_of_mass = self.dynamics.Evolution()
        self.pos1_hist.append(positions[0])
        self.pos2_hist.append(positions[1])
        self.pos3_hist.append(positions[2])
        
        pos1_hist_arr = np.array(self.pos1_hist)
        pos2_hist_arr = np.array(self.pos2_hist)
        pos3_hist_arr = np.array(self.pos3_hist)
        
        self.line1.set_data(pos1_hist_arr[:, 0], pos1_hist_arr[:, 1])
        self.line1.set_3d_properties(pos1_hist_arr[:, 2])
        
        self.line2.set_data(pos2_hist_arr[:, 0], pos2_hist_arr[:, 1])
        self.line2.set_3d_properties(pos2_hist_arr[:, 2])
        
        self.line3.set_data(pos3_hist_arr[:, 0], pos3_hist_arr[:, 1])
        self.line3.set_3d_properties(pos3_hist_arr[:, 2])

        self.body1.set_data(positions[0, 0], positions[0, 1])
        self.body1.set_3d_properties(positions[0, 2])
        
        self.body2.set_data(positions[1, 0], positions[1, 1])
        self.body2.set_3d_properties(positions[1, 2])
        
        self.body3.set_data(positions[2, 0], positions[2, 1])
        self.body3.set_3d_properties(positions[2, 2])
        
        self.center_of_mass_marker.set_data([center_of_mass[0]], [center_of_mass[1]])

        return self.line1, self.line2, self.line3, self.body1, self.body2, self.body3, self.center_of_mass_marker

    def start_animation(self):
        
        self.ani = FuncAnimation(self.fig, self.update, frames=iter(int, 1), cache_frame_data=False, init_func=self.init, blit=True, repeat=False)
        plt.legend()
        plt.show()
        
    
masses = [1e5, 3e5, 4e4]
tick = 0.001

# initial_positions3D = np.array([[3, 0, 0], [0, -3, -2], [-1, -1, -1]])
# initial_velocity3D = np.array([[-5,-5,-5],[-20,-5,5],[30,15,5]])
# planet_system = Dynamics(masses, initial_positions3D, initial_velocity3D, tick)
# animation = Animate3D(planet_system)
# animation.start_animation()

initial_positions2D = np.array([[7, 0, 0], [-3, 4, 0], [-5, -5, 0]])
initial_velocity2D = np.array([[-5,-5,0],[-20,-5,0],[30,20,0]])
planet_system = Dynamics(masses, initial_positions2D, initial_velocity2D, tick)
animation = Animate2D(planet_system)
animation.start_animation()


