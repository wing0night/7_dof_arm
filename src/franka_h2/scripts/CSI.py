import numpy as np

class ManualCubicSpline:
    def __init__(self, x, y):
        """
        手动实现三次样条插值
        :param x: 已知点x坐标（时间），要求严格递增
        :param y: 已知点y坐标（关节角度）
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n = len(x) - 1  # 区间数
        self.coeffs = []  # 存储各段多项式系数
        self._compute_coefficients()

    def _compute_coefficients(self):
        # ==== 步骤1：计算二阶导数（使用自然样条边界条件）====
        n = self.n
        h = np.diff(self.x)  # 区间长度h_i = x_{i+1} - x_i
        b = np.diff(self.y)/h  # δy_i / h_i

        # 构建三对角矩阵方程组
        A = np.zeros((n+1, n+1))
        rhs = np.zeros(n+1)

        # 内部节点方程（i=1到n-1）
        for i in range(1, n):
            A[i, i-1] = h[i-1]
            A[i, i] = 2*(h[i-1] + h[i])
            A[i, i+1] = h[i]
            rhs[i] = 3*(b[i] - b[i-1])

        # 自然样条边界条件（两端二阶导数为0）
        A[0, 0] = 1
        A[n, n] = 1

        # 解方程组得到二阶导数c_i
        c = np.linalg.solve(A, rhs)

        # ==== 步骤2：计算多项式系数 ====
        for i in range(n):
            # 当前区间参数
            dx = self.x[i+1] - self.x[i]
            dy = self.y[i+1] - self.y[i]

            # 计算各系数（基于Hermite插值公式）
            a = self.y[i]
            b_i = dy/dx - dx*(2*c[i] + c[i+1])/3
            d_i = (c[i+1] - c[i])/(3*dx)

            self.coeffs.append([a, b_i, c[i], d_i])

    def evaluate(self, t, der=0):
        """
        计算插值结果
        :param t: 输入时间点
        :param der: 导数阶数（0-位置，1-速度，2-加速度）
        """
        t = np.asarray(t)
        idx = np.searchsorted(self.x, t, side='right') - 1
        idx = np.clip(idx, 0, self.n-1)  # 处理边界

        results = []
        for i, ti in zip(idx, t):
            a, b, c, d = self.coeffs[i]
            dt = ti - self.x[i]
            
            if der == 0:  # 位置
                val = a + b*dt + c*dt**2 + d*dt**3
            elif der == 1:  # 速度
                val = b + 2*c*dt + 3*d*dt**2
            elif der == 2:  # 加速度
                val = 2*c + 6*d*dt
            else:
                raise ValueError("导数阶数必须是0、1或2")
            
            results.append(val)
        
        return np.array(results)