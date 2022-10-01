import pygame
import random as r
import neat

import os

class Spike:
    def __init__(self, size, x, y) -> None:
        self.rect = pygame.Rect(x, y, size, size)
        self.color = (255, 0, 0)


class Player():
    def __init__(self, size, x=0, y=0) -> None:
        self.size = size
        self.rect = pygame.Rect(x, y, size, size)
        self.xVel = 5
        self.yVel = 10
        self.mass = 1
        self.score = 0
        self.distanceFromSpike = 0
        
        self.color = (r.randint(0, 100), r.randint(0, 150), r.randint(0, 150))

        self.isJumping = False


    def jump(self):
        if self.isJumping:

            F = (0.5) * self.mass * (self.yVel ** 2)
            # print(F)
            self.rect[1] -= F
            self.yVel -= 1

            if self.yVel < 0:
                self.mass = -1
            
            if self.yVel == -11:
                self.isJumping = False
                self.yVel = 10
                self.mass = 1
                self.rect[1] += 5 


    def __repr__(self) -> str:
        return f"Player with rectangle {self.rect}"


class Game:
    def __init__(self, players = 1, spikeCount = 2) -> None:
        pygame.init()

        self.highScore = 0
        self.spikeCount = spikeCount
        self.window_width = 1500
        self.window_height = 300
        self.bg_color = [0, 179, 242]
        self.modifier = 1
        self.players = []

        self.spikes = []
        self.placeSpikes()

        self.WIN = pygame.display.set_mode((self.window_width, self.window_height))
        
        self.WIN.fill(self.bg_color)


    def main(self, genomes, config):
        nets = []
        ge = []

        playerSize = self.window_width/25

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            self.players.append(Player(size = playerSize, y = self.window_height - playerSize + 1))
            g.fitness = 0
            ge.append(g)


        run = True
        while run:
            pygame.time.delay(10)

            run = self.updateScreen(nets, ge)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        
        return self.highScore


    def updateScreen(self, nets, ge):
        if len(self.players) == 0:
            return False

        self.updateBGColor()
        self.WIN.fill(self.bg_color)
        self.updatePlayers(nets, ge)
        self.drawSpikes()

        pygame.display.update()

        return True


    def updateBGColor(self):
        
        if self.bg_color[1] > 244:
            self.modifier = -1
        elif self.bg_color[1] < 60:
            self.modifier = 1
            
        self.bg_color[1] += self.modifier
        

    def drawSpikes(self):
        for spike in self.spikes:
            pygame.draw.rect(self.WIN, spike.color, spike.rect)

    
    def updatePlayers(self, nets, ge):

        for g in ge:
            g.fitness += 0.05

        for x, player in enumerate(self.players):
            spikeDistances = []
            for spike in self.spikes:
                spikeDistances.append(abs(spike.rect.left - player.rect.right))

            player.distanceFromSpike = min(spikeDistances)

            for spike in self.spikes:
                if pygame.Rect.colliderect(player.rect, spike.rect):
                    if len(self.players) == 1:
                        self.highScore = player.score

                    ge[x].fitness -= 2

                    self.players.pop(x)
                    nets.pop(x)
                    ge.pop(x)


        for x, player in enumerate(self.players):
            a = player.distanceFromSpike
            b = player.xVel
            c = player.isJumping
            
            output = nets[self.players.index(player)].activate((a, b, c))
            
            if output[0] > 0.5:
                player.isJumping = True

            if player.isJumping:
                player.jump()

            player.rect[0] += player.xVel

            if player.rect.left  > self.window_width:
                player.rect[0] = -player.size

                ge[x].fitness += 5
                
                self.spike = self.placeSpikes()

            pygame.draw.rect(self.WIN, player.color, player.rect)

    
    def placeSpikes(self):
        spikeSize = self.window_width/40
        list = []
        xLoc = 0
        for i in range(self.spikeCount):
            if list.__len__() == 0:
                xLoc = r.randint(200, 500)
            else:
                xLoc = list[-1].rect[0] + r.randint(150, 300)
            list.append(Spike(size = spikeSize, x = xLoc, y = self.window_height - spikeSize + 5))
        self.spikes = list


game = Game(spikeCount = 5)

#inputs: distance from player to spike, player's xVel (potentially yVel)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(config)
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(game.main, 50)
    print(winner)


if __name__ == "__main__":

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
