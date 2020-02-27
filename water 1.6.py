import pygame
import numpy as np
import random
import math
import time
import noise

#stuff doesnt wanna go down or right :(

QUICKDRAW = True
CMULT = 1        #carve multiplier
FMULT = .01      #follow multiplier
RANDOMNESS = 0.5 #randomness of velocity when point is reset
R = 5            #radius each point can 'see' to follow
CR = 0           #radius each point carves a path with

PERLINWEIGHTS = [.7,.5,.25]

#CARVEWEIGHTS = np.array([[1]])

#Points follow landscape down until at height = 0, take velocity of mouse when made
#right click to reverse landscape
#see i think imma add perlin noise but lets be real is that actually gonna happen

class World():

    def __init__(self):

        self.size = (400,400)

        self.r = R
        self.genCarArr()
        #self.carvePattern = CARVEWEIGHTS
        self.cr = self.carvePattern.shape[0]//2

        self.screen = pygame.display.set_mode( self.size )

        self.genTerrArr()
        self.dots = []
        self.genWeights()

        self.ogArr = self.arr.copy()

        self.toMiddle = 1

    def genTerrArr(self):

        def genFun(x,y):
            return ( 1 - abs( x/self.size[0] - .5 ) )* 255
        def genFun2(x,y):
            return ( 255- math.hypot( x-self.size[0]/2, y-self.size[1]/2 )  )/1.2
        '''
        self.arr = np.zeros( self.size )
        for x in range( self.size[0] ):
            for y in range( self.size[1] ):
                self.arr[x,y] = genFun2(x,y)'''

        self.arr = noise.generate_perlin_noise_2d( self.size, ( 8,8 ) ) * PERLINWEIGHTS[0]
        self.arr+= noise.generate_perlin_noise_2d( self.size, (16,16) ) * PERLINWEIGHTS[1]
        self.arr+= noise.generate_perlin_noise_2d( self.size, (20,20) ) * PERLINWEIGHTS[2]

        self.arr += 1
        self.arr *= 20

        for x in range( self.size[0] ):
            for y in range( self.size[1] ):
                self.arr[x,y] += genFun2(x,y)

    def genCarArr(self):
        r = CR
        self.carvePattern = np.zeros( (2*r+1,2*r+1) )
        for x in range( 2*r+1 ):
            for y in range( 2*r+1 ):
                
                rx,ry = x-r, y-r
                self.carvePattern[x,y] = 1/math.hypot(rx,ry) if rx or ry else 1
        

    def randPos(self):

        return complex(random.randint(0,self.size[0]),random.randint(0,self.size[1]))

    def genWeights(self):
        r = self.r
        self.weights = np.zeros( (2*r+1,2*r+1), dtype = complex )
        for x in range( 2*r+1 ):
            for y in range( 2*r+1 ):
                rx,ry = x-r, y-r
                self.weights[x-1:x+1,y-1:y+1] = 1/(rx - 1j*ry) if rx or ry else 0

class Dot():

    def __init__( self, world, pos, vel=0 ):

        self.world = world
        self.reset( pos=pos )
        self.vel = vel

    def reset( self, pos=None ):
        self.pos = pos or self.world.randPos()
        self.dest = (200+200j)
        self.vel = complex((random.random()-.5)*RANDOMNESS,(random.random()-.5)*RANDOMNESS)

    def intPos(self):
        return int(self.pos.real), int(self.pos.imag)


                

    def update( self ):

        def out(self):
            x,y = self.intPos()
            if not 0<=x<self.world.size[0]-1 or not 0<=y<self.world.size[1]-1:
                self.reset()

        def finish( self ):

            x,y = self.intPos()
            if self.world.arr[ x-1, y-1 ] == 255:
                self.reset()
            if abs(self.vel)<.1:
                self.reset()

        def middle( self ):
            
            dis = self.pos - self.dest
            if abs(dis)==0:
                self.reset()
                return
            
            self.vel += - dis/abs(dis) * .0002 * self.world.toMiddle

        def drag( self ):
            self.vel -= abs(self.vel)*self.vel * .025

        def carve( self ):
            v = self.vel
            x,y = self.intPos()
            cr = self.world.cr
            carr = self.world.carvePattern
            if max( x,y ) <= 400-1-cr and min(x,y)>cr:
                
                self.world.arr[ x-cr:x+cr+1 , y-cr:y+cr+1 ] += carr * abs(v) * CMULT

        def follow(self):
            r = self.world.r
            x,y = int(self.pos.real), int(self.pos.imag)
            sx,sy = self.world.size
            if not r<x<sx-1-r or not r<y<sy-1-r:
                return
            tempArr = self.world.arr[x-r-1:x+r , y-r-1:y+r].copy()
            tempArr -= tempArr[r,r]
            tempArr = self.world.weights * tempArr
            
            self.vel += sum(sum(tempArr)) * FMULT / r**2
                    
        #forces =[ out, finish, middle, drag, follow, carve ]
        forces = [ out, finish, middle, drag, follow, carve ]

        for force in forces:
            force(self)



    def move( self ):

        self.pos += self.vel


def main():

    world = World()
    loop(world)
    pygame.quit()
    

def loop( world ):

    clock = pygame.time.Clock()
    startTime = time.time()

    while True:

        fps = clock.tick(120)

        #if time.time()>startTime+120:
        #    world.genArr()
        #    startTime = time.time()

        if events( world ):
            return

        for dot in world.dots:
            dot.update()

        for dot in world.dots:
            dot.move()

        draw( world )

        '''if np.amax(world.arr) >=256:
            world.arr *= 255/np.amax(world.arr)'''

        world.arr[world.arr>255] = 255
        world.arr[world.arr< 0 ] =  0

        #if random.random()<1/240:
        #    print(fps)


def events( world ):

    dmx,dmy = pygame.mouse.get_rel()
    mv = complex( dmx,dmy )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN:
            #if event.key == pygame.K_a:
            #    world.arr = blur(world.arr)
                
            if event.key == pygame.K_s:
                world.arr += world.ogArr
                world.arr /= 2

            if event.key == pygame.K_d:
                world.genTerrArr()
                world.dots = []

            if event.key == pygame.K_f:
                world.arr = (1-world.arr/255)**2 * 255

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[2]:
                world.arr = world.arr*-1 + 255
                world.toMiddle *= -1
                
            

        #if event.type == pygame.MOUSEBUTTONDOWN:
    if pygame.mouse.get_pressed()[0]:
        mx,my = pygame.mouse.get_pos()
        world.dots.append( Dot(world,complex(mx,my),vel=mv) )


def draw( world ):

    world.screen.fill( (0,255,255) )
    arrBlit( world.screen, world.arr, (0,0) )

    for d in world.dots:
        x,y = int( d.pos.real) , int(d.pos.imag)
        #pygame.draw.circle( world.screen, (255,255,0), (x,y), 1)
        pygame.draw.rect( world.screen, (255,0,0), (x,y,2,2) )
        #world.screen.set_at( (x,y) , (255,0,0) )

    pygame.display.update()


def arrBlit( surf, arr, pos ):

    sx1,sy1 = arr.shape
    narr = arr*-1+255

    surfarrayg = pygame.surfarray.pixels_green(surf)
    surfarrayr = pygame.surfarray.pixels_red(surf)

    surfarrayg[ pos[0]:, pos[1]: ] = narr
    surfarrayr[ pos[0]:, pos[1]: ] = narr
    del surfarrayg
    del surfarrayr

  
def blur(a):
    kernel = np.array([[1.0,2.0,1.0],
                       [2.0,4.0,2.0],
                       [1.0,2.0,1.0]])

    

    kernel = np.array([[ 0,-1,0 ],
                       [-1, 4,-1],
                       [ 0,-1,0 ]])
    kernel = kernel / np.sum(kernel)
    
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum


if __name__ == '__main__':
    main()
        
