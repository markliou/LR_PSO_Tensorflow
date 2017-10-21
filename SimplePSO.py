#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#   https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    #def evaluate(self,costFunc):
    def evaluate(self, costFunc, sess, bbatch_xs, bbatch_ys, fFeatureNo, lLabelNo):
        #self.err_i=costFunc(self.position_i)
        self.err_i = costFunc(self.position_i, sess, bbatch_xs, bbatch_ys, fFeatureNo, lLabelNo)[0]

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter,
                 sess, bbatch_xs, bbatch_ys, fFeatureNo, lLabelNo):
        global num_dimensions

        num_dimensions=len(x0)
        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                #swarm[j].evaluate(costFunc)
                swarm[j].evaluate(costFunc, sess, bbatch_xs, bbatch_ys, fFeatureNo, lLabelNo)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position(bounds)
            i+=1
            #print('iteration{} score:{}'.format(i, self.err_best_g))

        # print final results
        # print('FINAL:')
        # print(self.pos_best_g)
        # print(self.err_best_g)
        
    def results(self):
        return [self.err_best_g, self.pos_best_g]
    pass

        
def main():
    initial=[5 for i in range(1000)]               # initial starting location [x1,x2...]
    bounds=[(-10,10) for i in range(1000)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(func1,initial,bounds,num_particles=500,maxiter=5000)
pass

#--- RUN ----------------------------------------------------------------------+        
if __name__ == "__main__":
    main()





#--- END ----------------------------------------------------------------------+