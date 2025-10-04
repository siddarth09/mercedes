class MinimumCurvaturePlanner(Node):
    def __init__(self):
        super().__init__("minimum_curvature_planner")

        # ROS interfaces
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 1)
        self.path_pub = self.create_publisher(Path, "/raceline", 1)

        # Params
        self.declare_parameter('max_lateral_accel', 5.0)  # m/s²
        self.declare_parameter('max_longitudinal_accel', 3.0) 
        self.declare_parameter('max_decel', -6.0)
        self.declare_parameter('max_speed', 10.0)

        # Buffers
        self.centerline = None
        self.path = None
        self.speed_profile = None

    def map_callback(self, msg: OccupancyGrid):
        # 1. Extract track centerline from occupancy grid
        grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        binary = (grid == 0).astype(np.uint8)
        dist_map = distance_transform(binary)
        centerline_pts = self.extract_centerline(dist_map, msg.info)

        # 2. Fit cubic splines to get smooth x(t), y(t)
        spline_params = self.fit_splines(centerline_pts)

        # 3. Solve minimum curvature optimization (CasADi QP/NLP)
        raceline = self.solve_min_curvature(spline_params)

        # 4. Compute curvature κ(s) and velocity profile
        v_profile = self.forward_backward_pass(raceline)

        # 5. Publish as nav_msgs/Path with speed encoded
        path_msg = self.make_path_msg(raceline, v_profile, msg.header)
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published optimized raceline")

    def extract_centerline(self, dist_map, info):
        # Use skeletonization / distance ridge detection
        # Convert pixel to world coords
        return np.array([...])  # Nx2 array of [x,y]

    def fit_splines(self, pts):
        # Fit cubic splines piecewise
        return {...}

    def solve_min_curvature(self, splines):
        # Use CasADi to minimize ∑ κ² subject to continuity constraints
        return np.array([...])  # Nx2 raceline points

    def forward_backward_pass(self, raceline):
        # Compute curvature at each point
        # v_max = sqrt(a_lat_max / |κ|)
        # forward pass (accel limit), backward pass (brake limit)
        return np.array([...])  # velocities per point

    def make_path_msg(self, raceline, v_profile, header):
        path = Path()
        path.header = header
        for (x,y), v in zip(raceline, v_profile):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y
            # orientation: tangent of path
            path.poses.append(pose)
        return path
