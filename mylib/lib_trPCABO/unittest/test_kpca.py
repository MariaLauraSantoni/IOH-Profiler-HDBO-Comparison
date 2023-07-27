from bayes_optim.kpca import *
from bayes_optim.mylogging import eprintf
from bayes_optim.extension import KernelTransform, KernelFitStrategy, RealSpace
import pytest
import random


def test_kpca():
    random.seed(0)
    X = [[6.888437030500963, 5.159088058806049], [-1.5885683833831, -4.821664994140733], [0.22549442737217085, -1.9013172509917133], [5.675971780695452, -3.933745478421451], [-0.4680609169528829, 1.6676407891006235]]
    X_weighted = [[1.079483540452395, 0.808478123348775], [-0.0, -0.0], [0.015436285850211205, -0.13015521900151383], [2.8024450976474777, -1.9422411099521695], [-0.1315704411634408, 0.4687685435317057]]
    Y = [35.441907931514024, 455.983619954143, 288.22967496622755, 26.86758082381718, 30.021247428418974]
    kpca = MyKernelPCA(0.001, X, kernel_config={'kernel_name': 'rbf', 'kernel_parameters': {'gamma': 0.01}}, dimensions=2)
    space = RealSpace([-5, 5]) * 2
    kpca.enable_inverse_transform(space.bounds)
    kpca.fit(X_weighted)
    points = [[0.1, 4.3], [2.3, 3.2], [-1.2, 4.1], [0.97627008, 4.30378733]]
    ys = kpca.transform(points)
    # eprintf("transformed point is", ys)
    points1 = kpca.inverse_transform(ys)
    for (point, point1) in zip(points, points1):
        assert point == pytest.approx(point1, abs=0.1)


def test_poly_kernel():
    my_kernel = create_kernel('poly', {'gamma': 0.1,'d': 2,'c0': 0})
    assert my_kernel([1., 2.], [2., 3.]) == pytest.approx(0.64, abs=0.0001)


X = [[-3.5261636457420837, -2.351717984413572], [-0.7006502656326337, -4.566899291151256], [1.7729779622517245, 4.2261269157682655], [3.8998999798224556, -0.5553572234968236], [-2.729563653189096, 2.805196951058811], [4.832929712261667, 3.865893673749989], [3.9124322981507795, -4.971877090648892], [-0.1616436599443336, 0.11613330925482401], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [4.832929712261667, 3.865893673749989], [-4.867060615194007, 4.9945691484006876], [-1.9860986969950059, 3.2112837468619855], [-2.8723134299050783, 1.7676771760067167], [-1.4657214741455222, -3.692243533520105], [2.7708894654641236, 1.8538803354174291], [0.040778264675010334, 2.446797223084789], [-2.632665245745297, 2.3637546031531738], [-0.38638978135752566, 1.9949986329384561], [-4.999617881627624, -4.999350584219162], [-4.670272654869409, 1.7318585973129172], [0.2771155651877102, 3.5011898479365797], [-4.480953216322681, 2.893320596399356], [-3.425458722518958, 2.145437886908651], [-4.843765691130719, 4.855505489292614], [-1.341228249655169, 2.8575957979501982], [-2.784909143737248, 2.486004313421647], [-3.2770532373739147, 2.5342728135789905], [-3.5416175104634493, 2.6230400477720472], [-3.9900670493682697, 2.7739614707582874], [-3.9357543738566023, 2.9198451835015398], [-3.389817127225266, 2.4732174256575163], [4.832929712261667, 3.865893673749989], [3.677950467762358, 1.0628481656179396], [-0.0250796466799299, -2.0651154234823688]]
y = [321.7110225397448, 316.29726458888706, 328.42678691959094, 316.9146579917869, 314.3975168090511, 338.06466911306404, 346.78110744867587, 328.0456094847132, 338.06466911306404, 338.06466911306404, 338.06466911306404, 338.06466911306404, 338.06466911306404, 338.06466911306404, 338.06466911306404, 338.06466911306404, 340.36151758546544, 323.8493568258509, 318.59703202012486, 336.02696917646557, 329.04344351719294, 313.53375421752975, 310.76762898000925, 313.4668603767142, 350.3897234952767, 323.09153375827395, 326.6324090090402, 312.48703301265846, 312.6061648712787, 334.4643806393212, 330.5964322922658, 310.8022354634989, 310.63886785701175, 310.68250049026136, 311.11762013507064, 313.4170574206878, 310.6203091245215, 338.06466911306404, 313.95865305691444, 329.9710415133833]
transformed = [[-0.27653772073687344], [-0.03318296054571879], [0.06191568966851156], [0.2736486093314559], [-0.2953324620311276], [0.3037508899942051], [0.2902977231420611], [-0.055431712741943376], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [0.3037508899942051], [-0.4633453966737222], [-0.24042019878858817], [-0.29499868704403637], [-0.10265301525126605], [0.1667971039960614], [-0.06536659628618373], [-0.2832240104710734], [-0.09631681600916278], [-0.3106495531515253], [-0.4253673375832492], [-0.05601540692980545], [-0.4250330182142549], [-0.3414179641269439], [-0.461452336254931], [-0.18465122679484913], [-0.2963755951199938], [-0.33450780426847593], [-0.35518841693377196], [-0.38938320852578706], [-0.3868624622806487], [-0.3423368957011588], [0.3037508899942051], [0.24485706522124823], [-0.016439402695335236]]


def test_kernel_transform():
    kernel_transform = KernelTransform(2, minimize=True, kernel_fit_strategy=KernelFitStrategy.FIXED_KERNEL, kernel_config={'kernel_name': 'rbf', 'kernel_parameters': {'gamma': 0.002754228703338166}}, epsilon=0.2, X_initial_space=X)
    my_transformed = kernel_transform.fit_transform(X, y)
    for (my, right) in zip(my_transformed, transformed):
        assert my == pytest.approx(right, 0.001)


def test_kernel_parameter_fit():
    kernel_transform = KernelTransform(2, minimize=True, kernel_fit_strategy=KernelFitStrategy.LIST_OF_KERNEL, kernel_config={'kernel_names': ['rbf']}, epsilon=0.2, X_initial_space=X)
    best_gamma = 0.002754228703338166
    kernel_transform.fit_transform(X, y)
    my_gamma = kernel_transform.kernel_config['kernel_parameters']['gamma']
    assert my_gamma == pytest.approx(best_gamma, abs=0.01)
