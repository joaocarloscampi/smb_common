import numpy as np
import cv2

# ------------------------------------------------------------------------
# Testar se vamos propagar o mapa inteiro ou até o alvo - NÃO FUNCIONA

# Comprimir a imagem
# Verificar se o ponto está em um obstaculo
# ------------------------------------------------------------------------

# xMap = 6
# yMap = 6

def addNode(nodes, x, y, d):
    nodes.append([x, y, d])
    return nodes

def p(node):
    return node[1]*yMap + node[0]

def convertPixelToGazebo(xPixel, yPixel):
    x = 1 + xPixel*0.050000 * 2
    y = 29.186 - (1 + yPixel*0.050000 * 2)
    return x, y

# costmap = np.zeros((yMap,xMap))

# imgGray = cv2.imread('Dilated_Image.png')
#
# imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)

imgGray = cv2.imread("Dilated_Image.png")

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
    x = path[i][0]
    y = path[i][1]
    imgGray[y][x] = 150

print(convertPixelToGazebo(path[0][0], path[0][1]))

pathGazebo = []

for i in range(0, len(path), 10):
    x = path[i][0]
    y = path[i][1]
    imgGray[y][x] = 200

    xPath, yPath = convertPixelToGazebo(path[i][0], path[i][1])

    pathGazebo.append([xPath, yPath])

cv2.imshow("FUNCIONOU", imgGray)
cv2.waitKey(0)
