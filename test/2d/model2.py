from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF



class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 96, 128, 192, 256],
    ): 
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256, 128, 64,  32,  16,   8,   4
        #   3,  32, 64,  96, 128, 192, 256
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2d(feature*2, feature))

        self.bottleneck = DoubleConv2d(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class AIM(nn.Module):
    ''' 
    Image encoder with waypoint output and pid controller
    '''

    def __init__(self, config, device, in_channels=3, out_channels=3, features=[32, 64, 96, 128, 192, 256]):
        super().__init__()
        self.pred_len = config.pred_len
        self.config = config
        self.device = device

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.image_encoder = ImageCNN(512, normalize=True).to(self.device)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256, 128, 64,  32,  16,   8,   4
        #   3,  32, 64,  96, 128, 192, 256
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2d(feature*2, feature))

        self.bottleneck = DoubleConv2d(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=4, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)

    def forward(self, feature_emb, target_point):
    	'''
		Predicts future waypoints from image features and target point (goal location)
		Args:
			feature_emb (list): list of feature tensors
			target_point (tensor): goal location registered to ego-frame
    	'''
    	feature_emb = sum(feature_emb)
    	z = self.join(feature_emb)

    	output_wp = list()

    	# initial input variable to GRU
    	x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

    	# autoregressive generation of output waypoints
    	for _ in range(self.pred_len):
    		x_in = torch.cat([x, target_point], dim=1)
    		z = self.decoder(x_in, z)
    		dx = self.output(z)
    		x = dx + x
    		output_wp.append(x)

    	pred_wp = torch.stack(output_wp, dim=1)

    	return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' 
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata