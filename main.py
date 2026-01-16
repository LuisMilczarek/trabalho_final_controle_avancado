#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from pioneer import Pioneer
    

client = RemoteAPIClient()
sim = client.require('sim')

pioneer = Pioneer(sim)

def normalize_angle(input : float) -> float:
    return np.arctan2(np.sin(input), np.cos(input))

def create_path(type, start, end, amp=0.3, freq=np.pi):
    dist = np.linalg.norm(end-start)
    num_points = int(dist/0.1)
    if type == "LINE":
        return np.linspace(start, end, num_points)
    elif type == "SINE":
        t = np.linspace(0, dist, num_points)
        direction = (end - start) / dist
        perp = np.array([-direction[1], direction[0]])
        line_points = np.linspace(start, end, num_points)
        sine_wave = amp * np.sin(freq * t)
        return line_points + np.outer(sine_wave, perp)
    elif type == "RECTANGLE":
        p1 = start
        p2 = np.array([end[0], start[1]])
        p3 = end
        p4 = np.array([start[0], end[1]])
        points = []
        for s, e in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            d = np.linalg.norm(e-s)
            n = max(int(d/0.1), 1)
            points.append(np.linspace(s, e, n, endpoint=False))
        return np.vstack(points)
    elif type == "CIRCLE":
        radius = np.linalg.norm(start - end)
        start_angle = np.arctan2(start[1] - end[1], start[0] - end[0])
        circumference = 2 * np.pi * radius
        n = max(int(circumference / 0.1), 1)
        angles = np.linspace(start_angle, start_angle + 2 * np.pi, n)
        x = end[0] + radius * np.cos(angles)
        y = end[1] + radius * np.sin(angles)
        return np.vstack((x, y)).T
    raise NotImplementedError("Invalid path type")

class Controller(object):
    def __init__(self):
        # self.__threshold = 0.2
        self.target = np.array([0,0])
        self.gamma1 = 1.0
        self.gamma2 = 1.0
        self.h = 0.05
        # self.time_stamp = 0
        self.last_e = 0.0
        self.last_alpha = 0.0
    
    @property
    def target(self) -> np.ndarray:
        return self.__target
    
    @target.setter
    def target(self, pose : np.ndarray):
        if not isinstance(pose, np.ndarray) or pose.shape != (2,):
            raise ValueError("Invalid pose")
        self.__target = pose
    
    def getSpeeds(self, current_pose : np.ndarray, current_angle : float) -> np.ndarray:
        if not isinstance(current_pose, np.ndarray) or current_pose.shape != (2,):
            raise ValueError(f"Invalid pose: {current_pose}")
        e = np.linalg.norm(self.target-current_pose)
        phi = np.arctan2(self.target[1]-current_pose[1], self.target[0]-current_pose[0])
        alpha = normalize_angle(current_angle - phi)
        # dt = timestamp-self.time_stamp

        self.last_e = e
        self.last_alpha = alpha

        eta1 = self.gamma1 * e * np.cos(alpha)
        eta2 = 0
        if not np.isclose(alpha,0):
            eta2 = -self.gamma2 * alpha - self.gamma1 * np.cos(alpha) * (np.sin(alpha)/alpha) * (alpha- self.h * phi)
        # self.time_stamp = timestamp

        return np.array([eta1,eta2])



# sim.setStepping(True)

def main():
    # pioneer.setSpeeds()
    ct = Controller()
    sim.startSimulation()
    # pioneer.set_speed(0.0,np.pi/2)
    pose = np.array(pioneer.get_pose())

    # final_pose = np.array([0,0])
    final_pose = pose[:2]*-1
    # path = np.linspace(pose[:2], final_pose,int(np.linalg.norm(final_pose-pose[:2])/0.1))
    path_type = "RECTANGLE"
    path = create_path(path_type, pose[:2], final_pose)
    
    counter = 0
    ct.target = path[0]

    # Data logging
    log_time = []
    log_pose_x = []
    log_pose_y = []
    log_error_dist = []
    log_error_ang = []
    log_v = []
    log_w = []

    try:
        while counter < path.shape[0]:
            pose = pioneer.get_pose()
            current_time = sim.getSimulationTime()
            
            # print(pose)
            if np.linalg.norm(ct.target-np.array(pose[:2])) < 0.2:
                counter +=1
                if counter >= path.shape[0]:
                    break
                ct.target = path[counter]
                # print(f"Counter: {counter} from {path.shape[0]}")
            speeds = ct.getSpeeds(np.array(pose[:2]),pose[2])
            pioneer.set_speed(speeds[0],speeds[1])

            # Log data
            log_time.append(current_time)
            log_pose_x.append(pose[0])
            log_pose_y.append(pose[1])
            log_error_dist.append(ct.last_e)
            log_error_ang.append(ct.last_alpha)
            log_v.append(speeds[0])
            log_w.append(speeds[1])

    except KeyboardInterrupt:
        pass
        # sim.step()
    sim.stopSimulation()

    # Generate Plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Path Plot (Executed vs Desired)
    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'r--', label='Caminho de Referência')
    plt.plot(log_pose_x, log_pose_y, 'b-', label='Caminho Executado')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Rastreamento de Caminho: {path_type}')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_type}_path_{timestamp}.png')
    plt.close()

    # 2. Position vs Time
    plt.figure()
    plt.plot(log_time, log_pose_x, label='Posição X')
    plt.plot(log_time, log_pose_y, label='Posição Y')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Posição [m]')
    plt.title(f'Posição vs Tempo: {path_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_type}_position_time_{timestamp}.png')
    plt.close()

    # 3. Errors (Distance & Angle)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(log_time, log_error_dist, 'g')
    plt.ylabel('Erro de Distância [m]')
    plt.title(f'Erros de Rastreamento: {path_type}')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(log_time, log_error_ang, 'm')
    plt.ylabel('Erro Angular [rad]')
    plt.xlabel('Tempo [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{path_type}_errors_{timestamp}.png')
    plt.close()

    # 4. Control Signals
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(log_time, log_v, 'k')
    plt.ylabel('Vel. Linear [m/s]')
    plt.title(f'Sinais de Controle: {path_type}')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(log_time, log_w, 'c')
    plt.ylabel('Vel. Angular [rad/s]')
    plt.xlabel('Tempo [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{path_type}_controls_{timestamp}.png')
    plt.close()

    print(f"Simulação finalizada. Gráficos salvos com timestamp: {timestamp}")

if __name__ == "__main__":
    main()