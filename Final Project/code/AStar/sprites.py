import time
import random
import pygame
from threading import Thread
from settings import *
import util 


class BaseSprite(pygame.sprite.Sprite):
    """
    BaseSprite类 游戏中所有变化物体的底层父类
    """
    def __init__(self, image_name, screen):
        super().__init__()
        self.screen = screen
        self.direction = None
        self.speed = None
        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()

    def update(self):
        # 根据方向移动
        if self.direction == Settings.LEFT:
            self.rect.x -= self.speed
        elif self.direction == Settings.RIGHT:
            self.rect.x += self.speed
        elif self.direction == Settings.UP:
            self.rect.y -= self.speed
        elif self.direction == Settings.DOWN:
            self.rect.y += self.speed


class Bullet(BaseSprite):

    def __init__(self, image_name, screen):
        super().__init__(image_name, screen)
        self.speed = Settings.BULLET_SPEED


class TankSprite(BaseSprite):
    """
    ImageSprite类，BaseSprite的子类，所有带图片的精灵的父类
    """
    def __init__(self, image_name, screen):
        super().__init__(image_name, screen)
        self.type = None
        self.bullets = pygame.sprite.Group()
        self.is_alive = True
        self.is_moving = False

    def shot(self):
        """
        射击类，坦克调用该类发射子弹
        :return:
        """

        # 把消失的子弹移除
        self.__remove_sprites()
        if not self.is_alive:
            return
        if len(self.bullets) >= 3:
            return
        if self.type == Settings.HERO:
            pygame.mixer.music.load(Settings.FIRE_MUSIC)
            pygame.mixer.music.play()

        # 发射子弹
        bullet = Bullet(Settings.BULLET_IMAGE_NAME, self.screen)
        bullet.direction = self.direction
        if self.direction == Settings.LEFT:
            bullet.rect.right = self.rect.left
            bullet.rect.centery = self.rect.centery
        elif self.direction == Settings.RIGHT:
            bullet.rect.left = self.rect.right
            bullet.rect.centery = self.rect.centery
        elif self.direction == Settings.UP:
            bullet.rect.bottom = self.rect.top
            bullet.rect.centerx = self.rect.centerx
        elif self.direction == Settings.DOWN:
            bullet.rect.top = self.rect.bottom
            bullet.rect.centerx = self.rect.centerx
        self.bullets.add(bullet)

    def move_out_wall(self, wall):
        if self.direction == Settings.LEFT:
            self.rect.left = wall.rect.right + 2
        elif self.direction == Settings.RIGHT:
            self.rect.right = wall.rect.left - 2
        elif self.direction == Settings.UP:
            self.rect.top = wall.rect.bottom + 2
        elif self.direction == Settings.DOWN:
            self.rect.bottom = wall.rect.top - 2

    def __remove_sprites(self):
        """
        移除无用的子弹
        :return:
        """
        for bullet in self.bullets:
            if bullet.rect.bottom <= 0 or \
                    bullet.rect.top >= Settings.SCREEN_RECT.bottom or \
                    bullet.rect.right <= 0 or \
                    bullet.rect.left >= Settings.SCREEN_RECT.right:
                self.bullets.remove(bullet)
                bullet.kill()

    def update(self):
        if not self.is_alive:
            return
        super(TankSprite, self).update()

    def boom(self):
        pygame.mixer.music.load(Settings.BOOM_MUSIC)
        pygame.mixer.music.play()
        for boom in Settings.BOOMS:
            self.image = pygame.image.load(boom)
            time.sleep(0.05)
            self.screen.blit(self.image, self.rect)
        pygame.mixer.music.stop()
        super(TankSprite, self).kill()

    def kill(self):
        self.is_alive = False
        t = Thread(target=self.boom)
        t.start()


'Add Your Implement Here'

class Hero(TankSprite):

    def __init__(self, image_name, screen, tankWar):
        super(Hero, self).__init__(image_name, screen)
        self.tankWar = tankWar
        self.type = Settings.HERO
        self.speed = Settings.HERO_SPEED
        self.direction = Settings.UP
        self.is_hit_wall = False

        # 初始化英雄的位置
        self.rect.centerx = Settings.SCREEN_RECT.centerx - Settings.BOX_RECT.width * 2
        self.rect.bottom = Settings.SCREEN_RECT.bottom - 5

    def __turn(self):
        self.image = pygame.image.load(Settings.HERO_IMAGES.get(self.direction))

    def hit_wall(self):
        if self.direction == Settings.LEFT and self.rect.left <= 0 or \
                self.direction == Settings.RIGHT and self.rect.right >= Settings.SCREEN_RECT.right or \
                self.direction == Settings.UP and self.rect.top <= 0 or \
                self.direction == Settings.DOWN and self.rect.bottom >= Settings.SCREEN_RECT.bottom:
            self.is_hit_wall = True
        return self.is_hit_wall

    def getLegalAction(self):
        Map = Settings.MAP_ONE
        x, y = round(self.rect.centerx / Settings.BOX_SIZE - 0.5), round(self.rect.centery / Settings.BOX_SIZE - 0.5)    
        LegalAction = []
        if(x > 0 and Map[y][x-1] != 1 and Map[y][x-1] != 2 and Map[y][x-1] != 5):
            LegalAction.append(Settings.LEFT)
        if(x < 18 and Map[y][x+1] != 1 and Map[y][x+1] != 2 and Map[y][x+1] != 5):
            LegalAction.append(Settings.RIGHT)
        if(y > 0 and Map[y-1][x] != 1 and Map[y-1][x] != 2 and Map[y-1][x] != 5):
            LegalAction.append(Settings.UP)
        if(y != 12 and Map[y+1][x] != 1 and Map[y+1][x] != 2 and Map[y+1][x] != 5):
            LegalAction.append(Settings.DOWN)
        #print(LegalAction, x, y)
        return LegalAction

    def getAction(self):
        legalActions = self.getLegalAction()
        return max(legalActions, key= lambda x: self.evaluate(x))
        #return random.choice(legalActions)

    def evaluate(self, action):
        newPosition = (0, 0)
        enemies = self.tankWar.enemies
        bullets = []
        for enemy in enemies:
            for oneBullet in enemy.bullets:
                if(oneBullet.direction == Settings.UP):
                    bullets.append((oneBullet.rect.centerx, oneBullet.rect.centery-60))
                if(oneBullet.direction == Settings.DOWN):
                    bullets.append((oneBullet.rect.centerx, oneBullet.rect.centery+60))
                if(oneBullet.direction == Settings.LEFT):
                    bullets.append((oneBullet.rect.centerx-60, oneBullet.rect.centery))
                if(oneBullet.direction == Settings.RIGHT):
                    bullets.append((oneBullet.rect.centerx+60, oneBullet.rect.centery))

        disBullet = 2000
        if len(enemies) == 0:
            return 0

        if action == Settings.UP:
            newPosition = (self.rect.centerx, self.rect.centery-50)
        elif action == Settings.DOWN:
            newPosition = (self.rect.centerx, self.rect.centery+50)
        elif action == Settings.LEFT:
            newPosition = (self.rect.centerx-50, self.rect.centery)
        elif action == Settings.RIGHT:
            newPosition = (self.rect.centerx+50, self.rect.centery)
        #print("current position:", (self.rect.centerx, self.rect.centery),"action",action,"newPosition", newPosition)
        disEnemy = min([util.manhattanDistance(newPosition, oneEnemy.rect.center) for oneEnemy in enemies])
        
        if len(bullets) != 0:
            closetBullet = min(bullets, key= lambda x: util.manhattanDistance(newPosition, x))
            
            disBullet = util.manhattanDistance(newPosition, closetBullet)


        #print(action, disEnemy, disBullet)
        if disBullet <= 120:
            return -99999+disBullet
        elif disEnemy <= 100:
            return -9999+disEnemy
        else:
            return 1000-disEnemy

    def update(self):
        if not self.is_hit_wall:
            super().update()
            self.__turn()


    def getShoot(self):
        enemies = self.tankWar.enemies
        if self.direction == Settings.UP:
            for oneEneny in enemies:
                position = oneEneny.rect.center
                if(abs(self.rect.centerx - position[0]) <= 20 and position[1] < self.rect.centery):
                    return 1
        elif self.direction == Settings.DOWN:
            for oneEneny in enemies:
                position = oneEneny.rect.center
                if(abs(self.rect.centerx - position[0]) <= 20 and position[1] > self.rect.centery):
                    return 1
        elif self.direction == Settings.LEFT:
            for oneEneny in enemies:
                position = oneEneny.rect.center
                if(abs(self.rect.centery - position[1]) <= 20 and position[0] < self.rect.centerx):
                    return 1
        elif self.direction == Settings.RIGHT:
            for oneEneny in enemies:
                position = oneEneny.rect.center
                if(abs(self.rect.centery - position[1]) <= 20 and position[0] > self.rect.centerx):
                    return 1
        return 0

    def kill(self):
        self.is_alive = False
        self.boom()


class Enemy(TankSprite):

    def __init__(self, image_name, screen):
        super().__init__(image_name, screen)
        self.is_hit_wall = False
        self.type = Settings.ENEMY
        self.speed = Settings.ENEMY_SPEED
        self.direction = random.randint(0, 3)
        self.terminal = float(random.randint(40*2, 40*8))

    def random_turn(self):
        # 随机转向
        self.is_hit_wall = False
        directions = [i for i in range(4)]
        directions.remove(self.direction)
        self.direction = directions[random.randint(0, 2)]
        self.terminal = float(random.randint(40*2, 40*8))
        self.image = pygame.image.load(Settings.ENEMY_IMAGES.get(self.direction))

    def random_shot(self):
        shot_flag = random.choice(1*[True] + [False]*59)
        if shot_flag:
            super().shot()

    def hit_wall_turn(self):
        turn = False
        if self.direction == Settings.LEFT and self.rect.left <= 0:
            turn = True
            self.rect.left = 2
        elif self.direction == Settings.RIGHT and self.rect.right >= Settings.SCREEN_RECT.right-1:
            turn = True
            self.rect.right = Settings.SCREEN_RECT.right-2
        elif self.direction == Settings.UP and self.rect.top <= 0:
            turn = True
            self.rect.top = 2
        elif self.direction == Settings.DOWN and self.rect.bottom >= Settings.SCREEN_RECT.bottom-1:
            turn = True
            self.rect.bottom = Settings.SCREEN_RECT.bottom-2
        if turn:
            self.random_turn()

    def update(self):
        self.random_shot()
        if self.terminal <= 0:
            self.random_turn()
        else:
            super().update()
            # 碰墙掉头
            self.terminal -= self.speed


class Wall(BaseSprite):

    def __init__(self, image_name, screen):
        super().__init__(image_name, screen)
        self.type = None
        self.life = 2

    def update(self):
        pass

    '左上角的格子为(0,0)'
    def getPosition(self):
        return (self.rect.centerx/Settings.BOX_SIZE - 0.5, self.rect.centery/Settings.BOX_SIZE - 0.5)

    def boom(self):
        x, y = self.getPosition()
        Settings.MAP_ONE[int(y)][int(x)] = 0
        pygame.mixer.music.load(Settings.BOOM_MUSIC)
        pygame.mixer.music.play()
        for boom in Settings.BOOMS:
            self.image = pygame.image.load(boom)
            time.sleep(0.07)
            self.screen.blit(self.image, self.rect)
        pygame.mixer.music.stop()
        super().kill()

    def kill(self):
        self.life -= 1
        if not self.life:
            t = Thread(target=self.boom)
            t.start()

