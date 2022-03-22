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
import pandas as pd

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

    dist,ang = dist_and_angle(xPos,yPos,targetXPos,targetYPos)

    print('Distancia: ', dist)

    e = 0.5
    K = 5

    if objectPath.indexPath+1 < len(pathGazebo):

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
    else:
        publisherVel([0, 0, 0], [0, 0, 0])

    # Publisher para a posição do marker ser a posição do ponto mais próximo lido
    # publisherMarker(objectPath.path[objectPath.indexPath][0], objectPath.path[objectPath.indexPath][1], 0.8)

    #rospy.loginfo("Posição pilar: (%s, %s, %s)", xPos_pillar_robot, yPos_pillar_robot, zPos_pillar_robot)


def listener():

    #topic_name = rospy.get_param("/smb_highlevel_controller/topic_name")
    #topic_queue_size = rospy.get_param("/smb_highlevel_controller/topic_queue_size")

    topic_name = "odom"
    topic_queue_size = 10
    rospy.Subscriber(topic_name, Odometry, callbackScan, queue_size = topic_queue_size)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)

    file_map = rospy.get_param('~path_map')

    #pathGazebo = [[13.0, 5.385999999999999], [13.5, 5.385999999999999], [14.0, 4.885999999999999], [14.5, 4.785999999999998], [15.0, 4.785999999999998], [15.5, 4.785999999999998], [16.0, 4.785999999999998], [16.5, 4.785999999999998], [17.0, 4.785999999999998], [17.5, 4.785999999999998], [18.0, 4.785999999999998], [18.5, 4.785999999999998], [19.0, 4.785999999999998], [19.5, 4.785999999999998], [20.0, 4.785999999999998], [20.200000000000003, 3.8859999999999992], [20.200000000000003, 3.3859999999999992], [20.200000000000003, 2.8859999999999992], [20.200000000000003, 2.3859999999999992], [19.900000000000002, 2.0859999999999985], [19.400000000000002, 2.0859999999999985], [18.900000000000002, 2.0859999999999985], [18.400000000000002, 2.0859999999999985], [17.900000000000002, 2.0859999999999985], [17.400000000000002, 2.0859999999999985], [16.9, 2.0859999999999985], [16.4, 2.0859999999999985], [15.9, 2.0859999999999985], [15.4, 2.0859999999999985], [14.9, 2.0859999999999985], [14.4, 2.0859999999999985], [13.9, 2.0859999999999985], [13.4, 2.0859999999999985], [12.9, 2.0859999999999985], [12.4, 2.0859999999999985], [11.9, 2.0859999999999985], [11.4, 2.0859999999999985], [10.9, 2.0859999999999985], [10.4, 2.0859999999999985], [9.9, 2.0859999999999985], [9.700000000000001, 2.5859999999999985], [9.700000000000001, 3.0859999999999985], [9.700000000000001, 3.5859999999999985], [9.700000000000001, 4.0859999999999985], [9.700000000000001, 4.5859999999999985], [9.700000000000001, 5.0859999999999985], [9.700000000000001, 5.5859999999999985], [9.700000000000001, 6.0859999999999985], [9.700000000000001, 6.5859999999999985], [9.700000000000001, 7.0859999999999985], [9.700000000000001, 7.5859999999999985], [9.700000000000001, 8.085999999999999], [10.0, 9.085999999999999], [10.5, 9.085999999999999], [11.0, 9.085999999999999], [11.5, 9.085999999999999], [12.0, 9.686], [12.5, 9.686], [13.0, 10.186], [13.5, 10.686], [14.0, 11.186], [14.5, 11.485999999999997], [15.0, 11.485999999999997], [15.5, 11.485999999999997], [16.0, 11.485999999999997], [16.5, 11.485999999999997], [17.0, 11.485999999999997], [17.5, 11.485999999999997], [18.0, 11.485999999999997], [18.5, 11.485999999999997], [19.0, 11.485999999999997], [19.5, 11.485999999999997], [20.0, 11.485999999999997], [20.5, 11.585999999999999], [21.0, 12.085999999999999], [21.1, 12.585999999999999], [21.1, 12.585999999999999], [21.1, 13.085999999999999], [21.1, 13.585999999999999], [21.1, 14.085999999999999], [21.1, 14.585999999999999], [21.1, 15.085999999999999], [21.1, 15.585999999999999], [21.1, 16.086], [21.1, 16.586], [21.1, 17.086], [21.1, 17.586], [21.1, 18.086], [21.1, 18.586], [21.1, 19.086], [21.1, 19.586], [21.1, 20.086], [21.1, 20.586], [21.1, 21.086], [21.1, 21.586], [21.1, 22.086], [21.0, 22.586], [20.5, 23.186], [20.0, 23.186], [19.5, 23.186], [19.0, 23.186], [18.5, 22.286], [18.0, 21.786], [17.5, 21.286], [17.0, 20.786], [16.5, 20.286], [16.4, 19.786], [16.4, 19.286], [16.4, 19.286], [16.4, 18.785999999999998], [15.9, 18.086], [15.4, 18.086], [14.9, 18.086], [14.4, 18.086], [13.9, 18.086], [13.4, 18.086], [12.9, 18.086], [12.4, 18.086], [11.9, 18.086], [11.4, 18.086], [10.9, 18.086], [10.4, 18.086], [9.9, 18.086], [9.4, 18.886], [8.9, 18.886], [8.4, 19.386], [7.9, 19.886], [7.4, 20.386], [7.0, 20.886], [7.0, 21.386], [7.0, 21.886], [7.0, 22.386]]

    #pathGazebo = [[13.0, 6.0], [13.5, 5.5], [14.0, 5.0], [14.5, 4.899999999999999], [15.0, 4.899999999999999], [15.5, 4.899999999999999], [16.0, 4.899999999999999], [16.5, 4.899999999999999], [17.0, 4.899999999999999], [17.5, 4.899999999999999], [18.0, 4.899999999999999], [18.5, 4.899999999999999], [19.0, 4.899999999999999], [19.5, 4.899999999999999], [20.0, 4.899999999999999], [20.200000000000003, 4.5], [20.200000000000003, 4.0], [20.200000000000003, 3.5], [20.200000000000003, 3.0], [19.900000000000002, 2.6999999999999993], [19.400000000000002, 2.6999999999999993], [18.900000000000002, 2.6999999999999993], [18.400000000000002, 2.6999999999999993], [17.900000000000002, 2.6999999999999993], [17.400000000000002, 2.6999999999999993], [16.9, 2.6999999999999993], [16.4, 2.6999999999999993], [15.9, 2.6999999999999993], [15.4, 2.6999999999999993], [14.9, 2.6999999999999993], [14.4, 2.6999999999999993], [13.9, 2.6999999999999993], [13.4, 2.6999999999999993], [12.9, 2.6999999999999993], [12.4, 2.6999999999999993], [11.9, 2.6999999999999993], [11.4, 2.6999999999999993], [10.9, 2.6999999999999993], [10.4, 2.6999999999999993], [9.9, 2.6999999999999993], [9.700000000000001, 3.1999999999999993], [9.700000000000001, 3.6999999999999993], [9.700000000000001, 4.199999999999999], [9.700000000000001, 4.699999999999999], [9.700000000000001, 5.199999999999999], [9.700000000000001, 5.699999999999999], [9.700000000000001, 6.199999999999999], [9.700000000000001, 6.699999999999999], [9.700000000000001, 7.199999999999999], [9.700000000000001, 7.699999999999999], [9.700000000000001, 8.2], [9.700000000000001, 8.7], [10.0, 9.2], [10.5, 9.7], [11.0, 9.7], [11.5, 9.7], [12.0, 9.799999999999997], [12.5, 10.3], [13.0, 10.8], [13.5, 11.3], [14.0, 11.8], [14.5, 12.099999999999998], [15.0, 12.099999999999998], [15.5, 12.099999999999998], [16.0, 12.099999999999998], [16.5, 12.099999999999998], [17.0, 12.099999999999998], [17.5, 12.099999999999998], [18.0, 12.099999999999998], [18.5, 12.099999999999998], [19.0, 12.099999999999998], [19.5, 12.099999999999998], [20.0, 12.099999999999998], [20.5, 12.2], [21.0, 12.2], [21.1, 12.7], [21.1, 13.2], [21.1, 13.7], [21.1, 14.2], [21.1, 14.7], [21.1, 15.2], [21.1, 15.7], [21.1, 16.2], [21.1, 16.7], [21.1, 17.2], [21.1, 17.7], [21.1, 18.2], [21.1, 18.7], [21.1, 19.2], [21.1, 19.7], [21.1, 20.2], [21.1, 20.7], [21.1, 21.2], [21.1, 21.7], [21.1, 22.2], [21.1, 22.7], [21.0, 23.2], [20.5, 23.3], [20.0, 23.3], [19.5, 23.3], [19.0, 23.3], [18.5, 22.9], [18.0, 22.4], [17.5, 21.9], [17.0, 21.4], [16.5, 20.9], [16.4, 20.4], [16.4, 19.9], [16.4, 19.4], [16.4, 18.9], [15.9, 18.7], [15.4, 18.7], [14.9, 18.7], [14.4, 18.7], [13.9, 18.7], [13.4, 18.7], [12.9, 18.7], [12.4, 18.7], [11.9, 18.7], [11.4, 18.7], [10.9, 18.7], [10.4, 18.7], [9.9, 18.7], [9.4, 19.0], [8.9, 19.5], [8.4, 20.0], [7.9, 20.5], [7.4, 21.0], [7.0, 21.5], [7.0, 22.0], [7.0, 22.5], [7.0, 23.0]]


    df = pd.read_csv(file_map)

    tamanho = df.size

    pathGazebo = []

    for i in range(int(tamanho/2)):
        pathGazebo.append([df.iloc[i]['x'], df.iloc[i]['y']])

    print(pathGazebo)
    objectPath.getPath(pathGazebo)

    listener()
