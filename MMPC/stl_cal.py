class STL:
    def __init__(self, spec):
        self.spec = spec  # the root expression

    def evaluate(self, traj, time_grid=None):
        return self.spec.robustness(traj, time_grid or list(range(len(traj))))[0]

    class Expr:
        def robustness(self, traj, time_grid):
            raise NotImplementedError

    class Predicate(Expr):
        def __init__(self, func):
            self.func = func

        def robustness(self, traj, time_grid):
            return [self.func(state) for state in traj]

    class Not(Expr):
        def __init__(self, phi):
            self.phi = phi

        def robustness(self, traj, time_grid):
            return [-r for r in self.phi.robustness(traj, time_grid)]

    class And(Expr):
        def __init__(self, *args):
            self.sub = args

        def robustness(self, traj, time_grid):
            return [min(phi.robustness(traj, time_grid)[i] for phi in self.sub)
                    for i in range(len(traj))]

    class Or(Expr):
        def __init__(self, *args):
            self.sub = args

        def robustness(self, traj, time_grid):
            return [max(phi.robustness(traj, time_grid)[i] for phi in self.sub)
                    for i in range(len(traj))]

    class Globally(Expr):
        def __init__(self, phi, t1=0, t2=None):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2

        def robustness(self, traj, time_grid):
            r = self.phi.robustness(traj, time_grid)
            return [min(r[self.t1:(self.t2 or len(traj))])] * len(traj)

    class Eventually(Expr):
        def __init__(self, phi, t1=0, t2=None):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2

        def robustness(self, traj, time_grid):
            r = self.phi.robustness(traj, time_grid)
            return [max(r[self.t1:(self.t2 or len(traj))])] * len(traj)

    class Until(Expr):
        def __init__(self, phi1, phi2):
            self.phi1 = phi1
            self.phi2 = phi2

        def robustness(self, traj, time_grid):
            r1 = self.phi1.robustness(traj, time_grid)
            r2 = self.phi2.robustness(traj, time_grid)
            result = []
            for t in range(len(traj)):
                val = float('-inf')
                for k in range(t, len(traj)):
                    val = max(val, min(min(r1[t:k+1]), r2[k]))
                result.append(val)
            return result
