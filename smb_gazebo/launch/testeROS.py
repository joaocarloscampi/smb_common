#!/usr/bin/env python
# Importação de bibliotecas ROS
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import tf
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from geometry_msgs.msg import PoseWithCovarianceStamped

# Importação de bibliotecas relacionadas
import numpy as np
import cv2

class Path:
    def __init__(self):
        self.path = []
        self.lenPath = 0
        self.indexPath = 0

    def getPath(self, path):
        self.path = path
        self.lenPath = len(path)
        self.indexPath = 0

    def distPointPath(self, x, y):
        return np.sqrt( (x-self.path[self.indexPath][0])**2 + (y-self.path[self.indexPath][1])**2)

objectPath = Path()

# Criação do publisher de velocidade
pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
# Criação do publisher do Marker verde
pubMarker = rospy.Publisher('visualization_marker', Marker)

spawnX = 12.82
spawnY = 6.4125

# Função responsável por pegar a mínima distância lida pelo lidar
def minimun(data):
    vecRanges = data.ranges
    min = data.range_max
    index = 0
    for i in range(len(vecRanges)):
        if min > vecRanges[i]:
            min = vecRanges[i]
            index = i
    return index, min


# Função responsável por retornar as transformações do frame1 para o frame2
def get_tf(frame1, frame2):
    t = tf.TransformListener()
    try:
        (trans,rot) = t.lookupTransform(frame1, frame2, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("Erro no TF")

    return trans, rot


# Função responsável por publicar as velocidades (v, w) para o robô
def publisherVel(v, w):
    msg = Twist()
    msg.linear.x = v[0]
    msg.linear.y = v[1]
    msg.linear.z = v[2]
    msg.angular.x = w[0]
    msg.angular.y = w[1]
    msg.angular.z = w[2]
    pub.publish(msg)


# Função responsável por publicar o marcador utilizado para o cilíndro
def publisherMarker(x, y, z):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    pubMarker.publish(marker)


def addNode(nodes, x, y, d):
    nodes.append([x, y, d])
    return nodes


def p(node):
    return node[1]*yMap + node[0]


def convertPixelToGazebo(xPixel, yPixel):
    x = 1 + xPixel*0.05
    y = 30 - (1 + yPixel*0.05)
    return x, y

def dist_and_angle(x1,y1,x2,y2):
    dist = np.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = np.arctan2(y2-y1, x2-x1)
    return(dist,angle)

# Função responsável pelo callback de quando recebe uma mensagem LaserScan
def callbackScan(data):
    # index, min = minimun(data)
    #
    # # Resolução de leitura pelo sensor
    # angle_increment = data.angle_increment
    # # De qual angulo o sensor inicia a leitura
    # angle_min = data.angle_min
    #
    # # Angulo onde ocorre a menor distância lida
    # delta_angle = index*angle_increment
    # # Angulo com relação ao sensor que ocorre a leitura da menor distância
    # angle_pillar = delta_angle + angle_min
    #
    # # Posição (x, y, z) desse ponto mínimo com respeito ao sensor
    # xPos_pillar_sensor = min*np.cos(angle_pillar)
    # yPos_pillar_sensor = min*np.sin(angle_pillar)
    # zPos_pillar_sensor = 0
    #
    # # Obter a transformação do sensor para o base_link do robô
    # trans, rot = get_tf('rslidar', 'base_link')
    #
    # # Transformação de sistemas de coordenadas, do sensor para o robô
    # xPos_pillar_robot = xPos_pillar_sensor - trans[0]
    # yPos_pillar_robot = yPos_pillar_sensor - trans[1]
    # zPos_pillar_robot = zPos_pillar_sensor - trans[2]

    #print(data.pose.pose.position.x)


    '''xPos = spawnX + data.pose.pose.position.x
    yPos = spawnY + data.pose.pose.position.y

    targetXPos = objectPath.path[objectPath.indexPath][0]
    targetYPos = objectPath.path[objectPath.indexPath][1]'''

    # xPos = spawnX + data.pose.pose.position.x + 3
    # yPos = spawnY + data.pose.pose.position.y + 3
    xPos = data.pose.pose.position.x
    yPos = data.pose.pose.position.y

    targetXPos = objectPath.path[objectPath.indexPath][0]
    targetYPos = objectPath.path[objectPath.indexPath][1]

    print("xPos: ", xPos)
    print("yPos: ", yPos)
    print("targetXPos: ", targetXPos)
    print("targetYPos: ", targetYPos)

    angle = np.arctan2(targetYPos -yPos, targetXPos - xPos)

    print("angle: ", np.rad2deg(angle))

    orientation_list = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]

    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

    '''# if xPos > targetXPos:
    #     angle_1 = np.arctan2(np.sin(angle-yaw+180), np.cos(angle-yaw+180))
    # else:
    angle_1 = np.arctan2(np.sin(angle-yaw), np.cos(angle-yaw))

    print("angleZ: ", np.rad2deg(yaw))
    print("angle_1: ", np.rad2deg(angle_1))

    sin_angle_1 = np.sin(angle_1)

    print("sin angle_1: ", sin_angle_1)

    # Obtenção do ganho proporcional no parameter server
    #Kp = rospy.get_param("/smb_highlevel_controller/kp")
    Kp = 1

    # Equação do controlador P
    error = Kp*sin_angle_1

    # Publisher para a velocidade do robô após o controlador

    print("distancia: ", objectPath.distPointPath(xPos, yPos))
    if objectPath.distPointPath(xPos, yPos) < 0.2:
        #publisherVel([0, 0, 0], [0, 0, 0])
        if objectPath.indexPath+1 >= len(pathGazebo):
             publisherVel([0.0, 0, 0], [0, 0, 0])
        else:
            objectPath.indexPath = objectPath.indexPath + 1
    else:
        if abs(angle_1) > np.deg2rad(5):
            if error > 0:
                publisherVel([-0.3, 0, 0], [0, 0, 1]) # apenas 0.2
            else:
                publisherVel([-0.3, 0, 0], [0, 0, -1]) # apenas -0.2
        else:
            publisherVel([0.3, 0, 0], [0, 0, 0])'''

    dist,ang = dist_and_angle(xPos,yPos,targetXPos,targetYPos)

    print('Distancia: ', dist)

    e = 0.5
    K = 5

    if dist < e:
        objectPath.indexPath = objectPath.indexPath + 1
        dist,ang = dist_and_angle(xPos,yPos,targetXPos,targetYPos)

    e_ang = ang - yaw

    print('Erro angulo: ', e_ang)
    print('Caminho: ' , objectPath.indexPath)
    print('Orientação: ', yaw)
    print('Angulo do Alvo:', ang)

    if abs(e_ang) < np.deg2rad(5):
        publisherVel([0.5, 0, 0], [0, 0, 0.0])
    else:
        publisherVel([0, 0, 0], [0, 0, K*np.sin(e_ang)])

    # Publisher para a posição do marker ser a posição do ponto mais próximo lido
    # publisherMarker(objectPath.path[objectPath.indexPath][0], objectPath.path[objectPath.indexPath][1], 0.8)

    #rospy.loginfo("Posição pilar: (%s, %s, %s)", xPos_pillar_robot, yPos_pillar_robot, zPos_pillar_robot)


def listener():
    rospy.init_node('listener', anonymous=True)

    #topic_name = rospy.get_param("/smb_highlevel_controller/topic_name")
    #topic_queue_size = rospy.get_param("/smb_highlevel_controller/topic_queue_size")

    topic_name = "odom"
    topic_queue_size = 10
    rospy.Subscriber(topic_name, Odometry, callbackScan, queue_size = topic_queue_size)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    '''
    imgGray = cv2.imread("map.png")

    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)

    imgGray = cv2.resize(imgGray, (0, 0), fx = 0.5, fy = 0.5)

    cv2.imshow("Segmentada", imgGray)
    cv2.waitKey(0)

    (l1, l2) = imgGray.shape[:2]

    costmap = np.zeros((l1,l2))

    xMap = l2
    yMap = l1

    for i in range(l1):
        for j in range(l2):
            if imgGray[i][j] == 255:
                costmap[i][j] = 0
            else:
                costmap[i][j] = -1

    #print(costmap)

    # ------------------------------------------------

    # l2r = l2//4 + 1
    # l1r = l1r//4 + 1
    #
    # imagem = np.ones((l1r, l2r))*(-1)
    #
    # for i in range(0, l1, 4): # linha
    #     for j in range(0, l2, 4): # coluna
    #
    #         obstaculo = False
    #         for ii in range(i, i+4):
    #             for jj in range(j, j+4):
    #                 if costmap[jj][ii] == -1:
    #                     obstaculo = True
    #         if obstaculo:
    #             imagem no pixel = -1

    # --------------------------------------------------

    # padrão: [x, y, d]

    iteracao = 0

    # Define inicio
    # xStart = 50
    # yStart = 500

    xStart = 120
    yStart = 230

    # Define final
    # xEnd = 120
    # yEnd = 500

    xEnd = 60
    yEnd = 60

    # Cria vetor nó
    nodes = []

    # Adiciona final como primeiro no
    nodes = addNode(nodes, xEnd, yEnd, 1)

    print("Gerando o mapa... (Isso pode demorar um pouco)")

    # while que vai até acabar os nó
    while(len(nodes) > 0):
        #print("-----------Iteracao ", iteracao)
        # lista auxiliar pra armazenar novos nós
        new_nodes = []

        # verificar cada vizinho em cada no da onda
        for node in nodes:
            xNode = node[0]
            yNode = node[1]
            dNode = node[2]

            costmap[yNode][xNode] = dNode

            # verifica leste
            if ((xNode + 1) < xMap and costmap[yNode][xNode+1] == 0):
                new_nodes = addNode(new_nodes, xNode+1, yNode, dNode+1)

            # verifica oeste
            if ((xNode - 1) >= 0 and costmap[yNode][xNode-1] == 0):
                new_nodes = addNode(new_nodes, xNode-1, yNode, dNode+1)

            # verifica norte
            if ((yNode - 1) >= 0 and costmap[yNode-1][xNode] == 0):
                new_nodes = addNode(new_nodes, xNode, yNode-1, dNode+1)

            # verifica norte
            if ((yNode + 1) < yMap and costmap[yNode+1][xNode] == 0):
                new_nodes = addNode(new_nodes, xNode, yNode+1, dNode+1)

            for i in range (len(new_nodes)):
                for j in range(i+1, len(new_nodes)):
                    if p(new_nodes[i]) > p(new_nodes[j]):
                        new_nodes[i], new_nodes[j] = new_nodes[j], new_nodes[i]
            #print(new_nodes)


            index = 0

            while(index < len(new_nodes)-1):
                if p(new_nodes[index]) == p(new_nodes[index+1]):
                    new_nodes.pop(index+1)
                else:
                    index = index + 1

            #print(costmap)
            iteracao = iteracao + 1

            nodes = []
            nodes = new_nodes

    path = []

    path.append([xStart, yStart])

    nLocX = xStart
    nLocY = yStart

    bNoPath = False

    while( not (nLocX == xEnd and nLocY == yEnd) and not bNoPath):

        listVizinhos = []

        #print("Ponto: " +  str(nLocX) + " , " + str(nLocY))
        # 4-Way Connectivity
        # verifica norte
        if ((nLocY - 1) >= 0 and costmap[nLocY - 1][nLocX] > 0):
            listVizinhos = addNode(listVizinhos, nLocX, nLocY-1, costmap[nLocY-1][nLocX])

        # verifica sul
        if ((nLocY + 1) < yMap and costmap[nLocY + 1][nLocX] > 0):
            listVizinhos = addNode(listVizinhos, nLocX, nLocY+1, costmap[nLocY+1][nLocX])

        # verifica leste
        if ((nLocX + 1) < xMap and costmap[nLocY][nLocX+1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX+1, nLocY, costmap[nLocY][nLocX+1])

        # verifica oeste
        if ((nLocX - 1) >= 0 and costmap[nLocY ][nLocX-1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX-1, nLocY, costmap[nLocY][nLocX-1])

        # CASO QUEIRA TIRAR A VIZINHANÇA 8, COMENTAR EM BAIXO

        #verifica noroeste
        if ((nLocY -1) >= 0 and (nLocX -1) >= 0 and costmap[nLocY-1][nLocX-1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX-1, nLocY-1, costmap[nLocY-1][nLocX-1])

        #verifica nordeste
        if ((nLocY -1) >= 0 and (nLocX +1) < xMap and costmap[nLocY-1][nLocX+1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX+1, nLocY-1, costmap[nLocY-1][nLocX+1])

        #verifica sudoeste
        if ((nLocY +1) < yMap and (nLocX -1) >= 0 and costmap[nLocY+1][nLocX-1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX-1, nLocY+1, costmap[nLocY+1][nLocX-1])

        #verfica sudeste
        if ((nLocY +1) < yMap and (nLocX +1) < xMap and costmap[nLocY+1][nLocX+1] > 0):
            listVizinhos = addNode(listVizinhos, nLocX+1, nLocY+1, costmap[nLocY+1][nLocX+1])

        for i in range (len(listVizinhos)):
            for j in range(i+1, len(listVizinhos)):
                if listVizinhos[i][2] > listVizinhos[j][2]:
                    listVizinhos[i], listVizinhos[j] = listVizinhos[j], listVizinhos[i]

        if len(listVizinhos) == 0:
            bNoPath = True
        else:
            nLocX = listVizinhos[0][0]
            nLocY = listVizinhos[0][1]
            path.append([nLocX, nLocY]);



    for i in range(len(path)):
        print(path[i])
        if imgGray[path[i][1]+10][path[i][0]] == 0:
            print("entrei1")
            newY = path[i][1] - 5
            if imgGray[newY][path[i][0]] != 0:
                print("entrei2")
                path[i][1] = newY

    for i in range(len(path)):
        print(path[i])
        if imgGray[path[i][1]-10][path[i][0]] == 0:
            print("entrei1")
            newY = path[i][1] + 5
            if imgGray[newY][path[i][0]] != 0:
                print("entrei2")
                path[i][1] = newY



    for i in range(len(path)):
        x = path[i][0]
        y = path[i][1]
        imgGray[y][x] = 150

    print(convertPixelToGazebo(path[0][0], path[0][1]))



    pathGazebo = []

    for i in range(0, len(path), 5):
        x = path[i][0]
        y = path[i][1]
        imgGray[y][x] = 200

        xPath, yPath = convertPixelToGazebo(path[i][0], path[i][1])

        pathGazebo.append([xPath, yPath])

    print("Caminho gerado. Iniciando movimentação do robô")

    cv2.imshow("Segmentada", imgGray)
    cv2.waitKey(0)

    objectPath.getPath(pathGazebo)

    # '''

    # imgGray = cv2.imread("MapaFinal.png")
    #
    # imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
    #
    # path = []
    #
    # for i in range(len(imgGray)):
    #     for j in range(len(imgGray[i])):
    #         if imgGray[i][j] == 150:
    #             path.append([j, i])
    #
    # print(path[0])
    # pathGazebo = []
    #
    # for i in range(0, len(path), 10):
    #     x = path[i][0]
    #     y = path[i][1]
    #     #imgGray[y][x] = 200
    #
    #     xPath, yPath = convertPixelToGazebo(path[i][0], path[i][1])
    #
    #     pathGazebo.append([xPath, yPath])
    #
    # print("Caminho gerado. Iniciando movimentação do robô")

    #pathGazebo = [[13.0, 5.186], [14.0, 4.186], [15.0, 3.785999999999998], [16.0, 3.785999999999998], [17.0, 3.785999999999998], [18.0, 3.785999999999998], [19.0, 3.785999999999998], [19.900000000000002, 3.5859999999999985], [19.900000000000002, 2.5859999999999985], [19.2, 2.186], [18.2, 2.186], [17.2, 2.186], [16.200000000000003, 2.186], [15.200000000000001, 2.186], [14.200000000000001, 2.186], [13.200000000000001, 2.186], [12.200000000000001, 2.186], [11.200000000000001, 2.186], [10.200000000000001, 2.186], [10.0, 3.186], [10.0, 4.186], [10.0, 5.186], [10.0, 6.186], [10.0, 7.186], [10.4, 8.186], [11.4, 9.186], [12.4, 9.986], [13.4, 10.986], [14.4, 11.585999999999999], [15.4, 11.585999999999999], [16.4, 11.585999999999999], [17.400000000000002, 11.585999999999999], [18.400000000000002, 11.585999999999999], [19.400000000000002, 11.585999999999999], [20.400000000000002, 11.686], [20.8, 12.386], [20.8, 13.386], [20.8, 14.386], [20.8, 15.386], [20.8, 16.386], [20.8, 17.386], [20.8, 18.386], [20.8, 19.386], [20.8, 20.386], [20.8, 21.386], [20.400000000000002, 22.186], [19.400000000000002, 22.186], [18.400000000000002, 21.386], [17.400000000000002, 20.386], [16.4, 19.386], [16.1, 18.386], [15.100000000000001, 18.186], [14.100000000000001, 18.186], [13.100000000000001, 18.186], [12.100000000000001, 18.186], [11.100000000000001, 18.186], [10.1, 18.186], [9.1, 19.086], [8.100000000000001, 20.086], [7.1000000000000005, 21.086], [7.0, 22.086]]

    pathGazebo = [[13.0, 5.385999999999999], [13.5, 5.385999999999999], [14.0, 4.885999999999999], [14.5, 4.785999999999998], [15.0, 4.785999999999998], [15.5, 4.785999999999998], [16.0, 4.785999999999998], [16.5, 4.785999999999998], [17.0, 4.785999999999998], [17.5, 4.785999999999998], [18.0, 4.785999999999998], [18.5, 4.785999999999998], [19.0, 4.785999999999998], [19.5, 4.785999999999998], [20.0, 4.785999999999998], [20.200000000000003, 3.8859999999999992], [20.200000000000003, 3.3859999999999992], [20.200000000000003, 2.8859999999999992], [20.200000000000003, 2.3859999999999992], [19.900000000000002, 2.0859999999999985], [19.400000000000002, 2.0859999999999985], [18.900000000000002, 2.0859999999999985], [18.400000000000002, 2.0859999999999985], [17.900000000000002, 2.0859999999999985], [17.400000000000002, 2.0859999999999985], [16.9, 2.0859999999999985], [16.4, 2.0859999999999985], [15.9, 2.0859999999999985], [15.4, 2.0859999999999985], [14.9, 2.0859999999999985], [14.4, 2.0859999999999985], [13.9, 2.0859999999999985], [13.4, 2.0859999999999985], [12.9, 2.0859999999999985], [12.4, 2.0859999999999985], [11.9, 2.0859999999999985], [11.4, 2.0859999999999985], [10.9, 2.0859999999999985], [10.4, 2.0859999999999985], [9.9, 2.0859999999999985], [9.700000000000001, 2.5859999999999985], [9.700000000000001, 3.0859999999999985], [9.700000000000001, 3.5859999999999985], [9.700000000000001, 4.0859999999999985], [9.700000000000001, 4.5859999999999985], [9.700000000000001, 5.0859999999999985], [9.700000000000001, 5.5859999999999985], [9.700000000000001, 6.0859999999999985], [9.700000000000001, 6.5859999999999985], [9.700000000000001, 7.0859999999999985], [9.700000000000001, 7.5859999999999985], [9.700000000000001, 8.085999999999999], [10.0, 9.085999999999999], [10.5, 9.085999999999999], [11.0, 9.085999999999999], [11.5, 9.085999999999999], [12.0, 9.686], [12.5, 9.686], [13.0, 10.186], [13.5, 10.686], [14.0, 11.186], [14.5, 11.485999999999997], [15.0, 11.485999999999997], [15.5, 11.485999999999997], [16.0, 11.485999999999997], [16.5, 11.485999999999997], [17.0, 11.485999999999997], [17.5, 11.485999999999997], [18.0, 11.485999999999997], [18.5, 11.485999999999997], [19.0, 11.485999999999997], [19.5, 11.485999999999997], [20.0, 11.485999999999997], [20.5, 11.585999999999999], [21.0, 12.085999999999999], [21.1, 12.585999999999999], [21.1, 12.585999999999999], [21.1, 13.085999999999999], [21.1, 13.585999999999999], [21.1, 14.085999999999999], [21.1, 14.585999999999999], [21.1, 15.085999999999999], [21.1, 15.585999999999999], [21.1, 16.086], [21.1, 16.586], [21.1, 17.086], [21.1, 17.586], [21.1, 18.086], [21.1, 18.586], [21.1, 19.086], [21.1, 19.586], [21.1, 20.086], [21.1, 20.586], [21.1, 21.086], [21.1, 21.586], [21.1, 22.086], [21.0, 22.586], [20.5, 23.186], [20.0, 23.186], [19.5, 23.186], [19.0, 23.186], [18.5, 22.286], [18.0, 21.786], [17.5, 21.286], [17.0, 20.786], [16.5, 20.286], [16.4, 19.786], [16.4, 19.286], [16.4, 19.286], [16.4, 18.785999999999998], [15.9, 18.086], [15.4, 18.086], [14.9, 18.086], [14.4, 18.086], [13.9, 18.086], [13.4, 18.086], [12.9, 18.086], [12.4, 18.086], [11.9, 18.086], [11.4, 18.086], [10.9, 18.086], [10.4, 18.086], [9.9, 18.086], [9.4, 18.886], [8.9, 18.886], [8.4, 19.386], [7.9, 19.886], [7.4, 20.386], [7.0, 20.886], [7.0, 21.386], [7.0, 21.886], [7.0, 22.386]]

    print(pathGazebo)
    objectPath.getPath(pathGazebo)

    listener()
