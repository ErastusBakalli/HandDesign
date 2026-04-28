
import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

#  inicializimi i Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# koordinatat e kubit
def get_cube_vertices(size=1.0):
    s = size / 2
    return [
        [s, s, -s],
        [s, -s, -s],
        [-s, -s, -s],
        [-s, s, -s],
        [s, s, s],
        [s, -s, s],
        [-s, -s, s],
        [-s, s, s]
    ]

cube_surfaces = (
    (0, 1, 2, 3),
    (3, 2, 6, 7),
    (7, 6, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 6, 2),
    (4, 0, 3, 7)
)

cube_edges = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# 4. vizatimi i kubin
def draw_cube(center, size, face_color, edge_color):
    vertices = get_cube_vertices(size)
    vertices = [[v[0] + center[0], v[1] + center[1], v[2] + center[2]] for v in vertices]

    # faqet e kubit 
    glEnable(GL_LIGHTING)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (*face_color, 1.0))
    glBegin(GL_QUADS)
    for surface in cube_surfaces:
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

    # brinjet e kubit
    glDisable(GL_LIGHTING)
    glColor3fv(edge_color)
    glLineWidth(2)
    glBegin(GL_LINES)
    for edge in cube_edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    glEnable(GL_LIGHTING)

# inicializimi i kameres
cap = cv2.VideoCapture(0)

# inicializimi i pygame dhe OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -10)

# parametrat e kubit
cube_size = 1.0
cube_pos = [0.0, 0.0, 0.0]
rotation_angle = 0
face_color = (0.5, 0.5, 0.5)
edge_color = (1.0, 0.5, 0.0)

#  loop kryesor
running = True
while running:
    # kamera dhe fokusimi i dores
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        wrist = hand_landmarks.landmark[0]
        mid_base = hand_landmarks.landmark[9]

        # pozicioni i kubit
        cx = (wrist.x - 0.5) * 6
        cy = -(wrist.y - 0.5) * 4
        cube_pos = [cx, cy, 0]

        # rrotullimi i kubit sipas kycit te dores 
        vx = mid_base.x - wrist.x
        vy = mid_base.y - wrist.y
        angle_rad = math.atan2(vy, vx)
        rotation_angle = math.degrees(angle_rad)

        # madhesia e kubit sipas distaces se gishtit te madh dhe gishtit tregues 
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        distance = np.sqrt(dx*dx + dy*dy)
        cube_size = max(0.2, min(distance * 10, 2.5))

    # tregon kameren 
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        running = False

    # eventet e pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # vizatimi ne OpenGL
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glTranslatef(cube_pos[0], cube_pos[1], cube_pos[2])
    glRotatef(rotation_angle, 0, 0, 1)
    draw_cube([0, 0, 0], cube_size, face_color, edge_color)
    glPopMatrix()
    pygame.display.flip()
    pygame.time.wait(10)

# Mbyllja  <3
cap.release()
cv2.destroyAllWindows()
pygame.quit()
quit()
