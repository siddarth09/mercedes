#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import rclpy
from rclpy.node import Node
from functools import partial


class PIDSliderGUI(Node):
    def __init__(self):
        super().__init__('pid_slider_gui')

        self.node_name = 'pid_controller'
        self.param_names = ['Kp', 'Ki', 'Kd']

        # Declare parameters if not already declared
        self.declare_parameters('', [(name, 0.0) for name in self.param_names])

    def get_param_value(self, name):
        try:
            return self.get_parameter(name).get_parameter_value().double_value
        except:
            self.get_logger().warn(f"Parameter {name} not found")
            return 0.0

    def set_pid_param(self, name, value):
        self.set_parameters([rclpy.parameter.Parameter(name, rclpy.Parameter.Type.DOUBLE, value)])
        self.get_logger().info(f"Set {name} = {value}")

class PIDGUI(QWidget):
    def __init__(self, rcl_node):
        super().__init__()
        self.rcl_node = rcl_node
        self.setWindowTitle("PID Tuner")

        self.layout = QVBoxLayout()
        self.labels = {}
        self.sliders = {}

        for param in ['Kp', 'Ki', 'Kd']:
            initial_value = self.rcl_node.get_param_value(param)
            label = QLabel(f"{param}: {initial_value}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(int(initial_value * 100))
            slider.setSingleStep(1)
            slider.valueChanged.connect(partial(self.on_slider_change, param, label))


            self.layout.addWidget(label)
            self.layout.addWidget(slider)
            self.labels[param] = label
            self.sliders[param] = slider

        self.setLayout(self.layout)

    def on_slider_change(self, param, label, value):
        real_value = round(value / 100.0, 2)
        label.setText(f"{param}: {real_value}")
        self.rcl_node.set_pid_param(param, real_value)


def main():
    rclpy.init()
    rcl_node = PIDSliderGUI()

    app = QApplication(sys.argv)
    gui = PIDGUI(rcl_node)
    gui.show()

    # Let PyQt run in parallel with ROS2 spin
    from threading import Thread
    def spin_ros():
        rclpy.spin(rcl_node)
    Thread(target=spin_ros, daemon=True).start()

    app.exec_()
    rcl_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
