import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations

class Particle:
    #Physical atributes initialization

    def __init__(self, x, y, vx, vy, radius=0.01, design=None):

        self.pos = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius

        self.design = design
        if not self.design:
            #Set the ball design
            self.design = {'edgecolor': 'r', 'facecolor': 'r', 'fill': False}

    @property
    def x(self):
        return self.pos[0]
    @x.setter
    def x(self, value):
        self.pos[0] = value
    @property
    def y(self):
        return self.pos[1]
    @y.setter
    def y(self, value):
        self.pos[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlap(self, candidate):
        #Check colision

        return np.hypot(*(self.pos - candidate.pos)) < self.radius + candidate.radius

    def generate(self, ax):
        #Generate ball

        ball = Circle(xy=self.pos, radius=self.radius, **self.design)
        ax.add_patch(ball)
        return ball

    def advance_ball(self, dt):
        #Particle projection over time

        self.pos += self.v * dt   #  s = v * t

        #Particle-wall colision
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > 2:
            self.x = 2-self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1-self.radius
            self.vy = -self.vy

class Simulation:
    #Definição da caixa

    def __init__(self, n, radius=0.01, design=None):
        #Inicialize "n" number of particles
        self.init_particles(n, radius, design)

    def init_particles(self, n, radius, design=None):
        #Sets particle's atributes

        try:
            assert n == len(radius)
        except TypeError:
            def gen_radius(n, radius):
                for i in range(n):
                    yield radius
            radius = gen_radius(n, radius)

        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):
            while True:
                x, y = rad + (1 - 2*rad) * np.random.random(2)
               # x, y = rad + (1 - 2 * rad) * np.random.random(2)
                vr = 0.1 * np.random.random() + 0.05
                vtheta = 2*np.pi * np.random.random()
                vx, vy = vr * np.cos(vtheta), vr * np.sin(vtheta)
                particle = Particle(x, y, vx, vy, rad, design)
                #Avoid overlaped particles generation
                for p2 in self.particles:
                    if p2.overlap(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

    def collide(self):
        #Elastic colision

        def change_velocity(p1, p2):

            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.pos, p2.pos
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2

       #Test colision between different particles
        pairs = combinations(range(self.n), 2)
        for i,j in pairs:
            if self.particles[i].overlap(self.particles[j]):
                change_velocity(self.particles[i], self.particles[j])

    def continue_animation(self, dt):
        #Animation projection over time
        for i, p in enumerate(self.particles):
            p.advance_ball(dt)
            self.balls[i].center = p.pos
        self.collide()
        return self.balls

    def update_position(self, dt):
        for i, p in enumerate(self.particles):
            p.advance_ball(dt)
        self.collide()

    def init(self):
        #Initialize animation

        self.balls = []
        for particles in self.particles:
            self.balls.append(particles.generate(self.ax))
        return self.balls

    def animate(self, i):
        #Continuous animation execution

        self.continue_animation(0.01)
        return self.balls


    def box(self, save=False):
        #Set box

        fig, self.ax = plt.subplots()
        for s in ['top','bottom','left','right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                               frames=400, interval=0.01, blit=True)
        plt.show()

if __name__ == '__main__':
    nparticles = 3
    cradius = np.random.random(nparticles)*0.3+0.01
    design = {'edgecolor': 'r','facecolor': 'r',  'linewidth': 1, 'fill': True}
    simulate = Simulation(nparticles, cradius, design)
    simulate.box(save=False)