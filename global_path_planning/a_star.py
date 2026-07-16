import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =========================
# Node
# =========================

class Node:

    def __init__(self, parent=None, position=None):

        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0


    def __eq__(self, other):

        return self.position == other.position



# =========================
# A* generator
# =========================

def astar_generator(grid, start, end):


    start_node = Node(None, start)
    end_node = Node(None, end)


    open_list = [start_node]

    closed_list = set()



    directions = [

        (-1,0),
        (1,0),
        (0,-1),
        (0,1),

        (-1,-1),
        (-1,1),
        (1,-1),
        (1,1)

    ]



    while open_list:



        current_node = min(
            open_list,
            key=lambda x:x.f
        )


        open_list.remove(
            current_node
        )


        closed_list.add(
            current_node.position
        )



        # 找到终点

        if current_node == end_node:


            path=[]

            cur=current_node


            while cur:

                path.append(
                    cur.position
                )

                cur=cur.parent



            yield (

                path[::-1],
                [n.position for n in open_list],
                list(closed_list),
                True

            )

            return



        # 搜索邻居

        for dx,dy in directions:


            nx=current_node.position[0]+dx

            ny=current_node.position[1]+dy



            if nx<0 or nx>=grid.shape[0]:
                continue


            if ny<0 or ny>=grid.shape[1]:
                continue



            # 障碍

            if grid[nx,ny]==1:
                continue



            if (nx,ny) in closed_list:
                continue



            child=Node(
                current_node,
                (nx,ny)
            )



            # g

            child.g = (
                current_node.g
                +
                np.sqrt(dx*dx+dy*dy)
            )


            # h

            child.h = (

                abs(nx-end[0])
                +
                abs(ny-end[1])

            )


            child.f = child.g + child.h



            skip=False


            for node in open_list:


                if (
                    node==child
                    and
                    child.g>=node.g
                ):

                    skip=True
                    break



            if not skip:

                open_list.append(
                    child
                )



        # 每一步刷新

        yield (

            None,

            [n.position for n in open_list],

            list(closed_list),

            False

        )



# ==================================================
# 读取地图
# ==================================================

data=np.load(
    "orchard_scene.npz",
    allow_pickle=True
)


occ=data["occ"].astype(int)


start=tuple(
    data["start"]
)


dyn=data["dyn"]



h,w=occ.shape



print("地图:",occ.shape)

print("起点:",start)



# 当前目标

goal=None


# 当前A*

generator=None



# ==================================================
# matplotlib
# ==================================================

fig,ax=plt.subplots(
    figsize=(8,8)
)



ax.set_xticks(
    np.arange(-0.5,w,1),
    minor=True
)


ax.set_yticks(
    np.arange(-0.5,h,1),
    minor=True
)


ax.grid(
    which="minor",
    color="gray",
    linewidth=0.3
)


ax.tick_params(
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False
)



# 初始图像

show=np.ones(
    (h,w,3)
)


show[occ==1]=[
    0,
    0,
    0
]


im=ax.imshow(
    show,
    origin="upper"
)



# ==================================================
# 鼠标点击重新规划
# ==================================================

def onclick(event):

    global goal
    global generator



    if event.xdata is None or event.ydata is None:

        return



    # matplotlib坐标

    x=int(round(event.xdata))

    y=int(round(event.ydata))



    # 转矩阵坐标

    new_goal=(y,x)



    # 越界

    if not(
        0<=new_goal[0]<h
        and
        0<=new_goal[1]<w
    ):

        return



    # 障碍

    if occ[new_goal]:

        print(
            "目标点为障碍:",
            new_goal
        )

        return



    goal=new_goal



    print(
        "重新规划:",
        goal
    )



    generator=astar_generator(

        occ,

        start,

        goal

    )



    ax.set_title(
        f"A* searching -> {goal}"
    )



fig.canvas.mpl_connect(
    'button_press_event',
    onclick
)



# ==================================================
# 动画刷新
# ==================================================

def update(frame):


    global generator



    img=np.ones(
        (h,w,3)
    )



    # 障碍

    img[occ==1]=[
        0,
        0,
        0
    ]



    # 没有目标

    if generator is None:


        img[start]=[
            1,
            0,
            0
        ]


        im.set_array(img)

        return [im]



    try:


        path,open_pts,closed_pts,done=next(generator)



        # closed

        for x,y in closed_pts:

            img[x,y]=[
                0.2,
                0.4,
                0.8
            ]



        # open

        for x,y in open_pts:

            img[x,y]=[
                0.5,
                0.8,
                1
            ]



        # 动态障碍

        for d in dyn:

            x=int(d[0])

            y=int(d[1])


            img[x,y]=[
                0,
                0,
                1
            ]



        # 起点

        img[start]=[
            1,
            0,
            0
        ]



        # 目标

        if goal is not None:

            img[goal]=[
                0,
                1,
                0
            ]



        # 路径

        if path is not None:


            for p in path:

                if p!=start and p!=goal:

                    img[p]=[
                        1,
                        0.8,
                        0
                    ]


            ax.set_title(
                "A* Path Found"
            )



        im.set_array(
            img
        )


    except StopIteration:

        pass



    return [im]



ani=animation.FuncAnimation(

    fig,

    update,

    interval=50,

    blit=True

)



ax.set_title(
    "Click map to plan path"
)



plt.show()