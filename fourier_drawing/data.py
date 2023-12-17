"""Data Module.
    The Data module provides functions to load the different data points.
    
    implements following functions:

    - load_data
    - normalize
    - _load_data_from_csv
    - _load_rabbit
    - _load_ellipse
    - _load_line
"""

import json
from pathlib import Path
from typing import List
import numpy as np


def load_data(dataset="rabbit") -> List[List[float]]:
    """load data either from sub function of via json

    Args:
        dataset (str, optional): Name of dataset. Defaults to "rabbit".

    Raises:
        FileNotFoundError: if dataset as json does not exists.
        ValueError: if dataset is not correctly formatted.

    Returns:
        List[List[float]]: returns coordinates as nested Lists.
    """

    match dataset:
        case "rabbit":
            data = _load_rabbit()
        case "ellipse":
            data = _load_ellipse()
        case "line":
            data = _load_line()
        case _:
            try:
                data = _load_data_from_json(dataset)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    (
                        f"No dataset file {dataset} found in data-dir. "
                        "Consider Creation at project-root/data."
                    )
                ) from exc
    if len(data[0]) != 2:
        raise ValueError(
            "Dataset is not correctly formatted. Check README for instruction."
        )
    return data


def normalize(point_array: np.ndarray, min_v=-1, max_v=1) -> np.ndarray:
    """_summary_

    Args:
        point_array (np.ndarray): _description_
        min_v (int, optional): _description_. Defaults to -1.
        max_v (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    x = point_array[:, 0]
    y = point_array[:, 1]
    x_norm = (max_v - min_v) * ((x - x.min()) / (x.max() - x.min())) + min_v
    y_norm = (max_v - min_v) * ((y - y.min()) / (y.max() - y.min())) + min_v
    return np.array([x_norm, y_norm]).T


def _load_data_from_json(dataset):
    dataset_path = Path(f"data/{dataset}").with_suffix(".json")
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def _load_rabbit():
    return np.array(
        [
            [591.7699890136719, 329.3479919433594],
            [573.8238677978516, 302.8184585571289],
            [553.2171783447266, 278.2467803955078],
            [531.2021331787109, 254.92733764648438],
            [507.4439697265625, 233.4050064086914],
            [495.7383117675781, 210.3707733154297],
            [499.9385681152344, 178.6180076599121],
            [498.18341064453125, 146.64281463623047],
            [489.5765838623047, 115.82509422302246],
            [473.2870178222656, 88.32209205627441],
            [450.70013427734375, 85.84137344360352],
            [433.9762420654297, 113.09134674072266],
            [424.64459228515625, 143.7001495361328],
            [422.1361846923828, 175.62327194213867],
            [425.6429748535156, 207.46001434326172],
            [415.3757019042969, 218.6624526977539],
            [405.14898681640625, 188.30846405029297],
            [389.58319091796875, 160.31619262695312],
            [368.8749465942383, 135.903564453125],
            [343.02933502197266, 117.06257247924805],
            [312.95423889160156, 106.33516120910645],
            [305.28285217285156, 133.94140625],
            [309.27569580078125, 165.6783447265625],
            [320.6389923095703, 195.60313415527344],
            [337.77306365966797, 222.66282653808594],
            [359.5541687011719, 246.1480484008789],
            [379.2557678222656, 267.4534454345703],
            [357.7966995239258, 263.62903594970703],
            [328.8069152832031, 249.99008178710938],
            [298.00154876708984, 241.17941284179688],
            [266.1994934082031, 237.260498046875],
            [234.16905975341797, 238.18577575683594],
            [202.65435409545898, 243.97554779052734],
            [172.4437599182129, 254.64627075195312],
            [144.37926483154297, 270.0970993041992],
            [119.37139511108398, 290.1146697998047],
            [98.41934585571289, 314.33670806884766],
            [81.36497688293457, 338.30123138427734],
            [52.28910446166992, 344.1581344604492],
            [49.826026916503906, 374.0392532348633],
            [68.9356632232666, 392.98453521728516],
            [65.92932605743408, 424.8849639892578],
            [65.40990734100342, 456.94920349121094],
            [67.37814521789551, 488.95196533203125],
            [73.17794799804688, 520.4635467529297],
            [86.29299736022949, 549.5435333251953],
            [112.8403491973877, 565.7662200927734],
            [144.9057846069336, 566],
            [176.9799346923828, 566],
            [209.05391311645508, 566],
            [241.12866973876953, 566],
            [273.2029342651367, 566],
            [301.5389633178711, 562.5301971435547],
            [281.7184143066406, 538.9510192871094],
            [251.14957427978516, 529.6452789306641],
            [219.21149444580078, 527.1273498535156],
            [244.14400482177734, 512.5182342529297],
            [267.45992279052734, 490.6217956542969],
            [284.38905334472656, 463.486572265625],
            [293.7195129394531, 432.8966064453125],
            [294.8676071166992, 400.9315643310547],
            [287.8056182861328, 369.73783111572266],
            [272.92372131347656, 341.42864990234375],
            [251.27751922607422, 317.8825454711914],
            [224.33045959472656, 300.6520690917969],
            [193.85908889770508, 290.93370056152344],
            [200.42051315307617, 289.3451919555664],
            [231.71697998046875, 295.9360046386719],
            [260.2548370361328, 310.3767776489258],
            [284.1445846557617, 331.6441116333008],
            [301.8129348754883, 358.3056869506836],
            [312.02884674072266, 388.6140365600586],
            [314.14270782470703, 420.5322723388672],
            [308.30116271972656, 451.9777526855469],
            [294.0954284667969, 480.6088562011719],
            [272.26818084716797, 503.9588317871094],
            [244.9180145263672, 520.5431976318359],
            [266.58099365234375, 524.5122833251953],
            [297.2285919189453, 533.5632781982422],
            [319.5755157470703, 555.4964752197266],
            [348.32518005371094, 545.2305755615234],
            [375.54895782470703, 528.3436737060547],
            [399.4787826538086, 507.0485534667969],
            [420.65362548828125, 523.2336120605469],
            [441.7372589111328, 547.4042663574219],
            [465.3560791015625, 566],
            [497.43162536621094, 566],
            [529.5061645507812, 566],
            [519.7122650146484, 537.7205505371094],
            [489.1353454589844, 529.4800262451172],
            [474.64625549316406, 508.53033447265625],
            [487.29005432128906, 483.9589080810547],
            [506.1493225097656, 458.216796875],
            [515.4617004394531, 427.66221618652344],
            [515.7876739501953, 395.6822738647461],
            [537.9475555419922, 384.6746368408203],
            [567.5895843505859, 372.7527770996094],
            [590.4958953857422, 350.8195114135742],
            [512.3771514892578, 318.1249771118164],
            [525.6879577636719, 299.36785888671875],
        ]
    )


def _load_ellipse():
    return np.array(
        [
            [393.6622314453125, 485.0917053222656],
            [393.8092346191406, 496.25213623046875],
            [394.2546081542969, 507.4046630859375],
            [395.00103759765625, 518.5410766601562],
            [396.0528869628906, 529.6528015136719],
            [397.4147033691406, 540.7307739257812],
            [399.0928039550781, 551.7652587890625],
            [401.09503173828125, 562.7455444335938],
            [403.4318542480469, 573.659423828125],
            [406.1138916015625, 584.49365234375],
            [409.15631103515625, 595.232177734375],
            [412.57513427734375, 605.8568115234375],
            [416.3908996582031, 616.3452758789062],
            [420.6260986328125, 626.6714477539062],
            [425.30804443359375, 636.8026733398438],
            [430.46881103515625, 646.6983032226562],
            [436.1439514160156, 656.3079833984375],
            [442.3748474121094, 665.5665283203125],
            [449.2046813964844, 674.3917846679688],
            [456.67913818359375, 682.6773681640625],
            [464.84063720703125, 690.2855224609375],
            [473.7224426269531, 697.0365600585938],
            [483.328125, 702.7069091796875],
            [493.6041259765625, 707.0401000976562],
            [504.4131774902344, 709.776611328125],
            [515.5222473144531, 710.7146606445312],
            [526.63134765625, 709.776611328125],
            [537.4403686523438, 707.0401000976562],
            [547.7164306640625, 702.7071533203125],
            [557.3218383789062, 697.036376953125],
            [566.2039794921875, 690.2857666015625],
            [574.3653564453125, 682.6774291992188],
            [581.83984375, 674.3920288085938],
            [588.6697387695312, 665.5667114257812],
            [594.9002075195312, 656.3079833984375],
            [600.5755615234375, 646.698486328125],
            [605.7362060546875, 636.802734375],
            [610.4185791015625, 626.6716918945312],
            [614.6537475585938, 616.3455200195312],
            [618.4691162109375, 605.8569946289062],
            [621.8883666992188, 595.2325439453125],
            [624.930419921875, 584.4939575195312],
            [627.6126098632812, 573.6597290039062],
            [629.9492797851562, 562.7456665039062],
            [631.9518432617188, 551.765380859375],
            [633.6299438476562, 540.7308959960938],
            [634.9915771484375, 529.6528015136719],
            [636.0431518554688, 518.5410766601562],
            [636.7901000976562, 507.4046936035156],
            [637.2353515625, 496.25213623046875],
            [637.38232421875, 485.0915832519531],
            [637.2353515625, 473.93109130859375],
            [636.7899169921875, 462.77862548828125],
            [636.0435180664062, 451.6423034667969],
            [634.99169921875, 440.5306396484375],
            [633.6298217773438, 429.4526062011719],
            [631.9517211914062, 418.418212890625],
            [629.9495239257812, 407.4378662109375],
            [627.6126708984375, 396.5238342285156],
            [624.9306030273438, 385.68963623046875],
            [621.88818359375, 374.9510803222656],
            [618.4693603515625, 364.32647705078125],
            [614.653564453125, 353.8377990722656],
            [610.4183349609375, 343.5116882324219],
            [605.736328125, 333.3804626464844],
            [600.5755615234375, 323.4847412109375],
            [594.9004516601562, 313.8752136230469],
            [588.6695556640625, 304.61669921875],
            [581.839599609375, 295.791259765625],
            [574.365234375, 287.5058898925781],
            [566.2037353515625, 279.8977355957031],
            [557.3220825195312, 273.1468963623047],
            [547.7162475585938, 267.47645568847656],
            [537.4403686523438, 263.1432800292969],
            [526.6314392089844, 260.4068603515625],
            [515.5224609375, 259.46875],
            [504.4132995605469, 260.4068298339844],
            [493.60418701171875, 263.14329528808594],
            [483.328125, 267.4763488769531],
            [473.72265625, 273.1471252441406],
            [464.84051513671875, 279.8977355957031],
            [456.6791687011719, 287.5060729980469],
            [449.2047119140625, 295.7915344238281],
            [442.3748474121094, 304.61676025390625],
            [436.14434814453125, 313.87548828125],
            [430.4690246582031, 323.48492431640625],
            [425.308349609375, 333.38067626953125],
            [420.62603759765625, 343.51171875],
            [416.3907775878906, 353.8380432128906],
            [412.5753479003906, 364.32659912109375],
            [409.1561279296875, 374.95111083984375],
            [406.1141052246094, 385.6895446777344],
            [403.4319763183594, 396.5235595703125],
            [401.0953369140625, 407.4375305175781],
            [399.0927429199219, 418.41754150390625],
            [397.41461181640625, 429.4520568847656],
            [396.0530700683594, 440.52984619140625],
            [395.0014343261719, 451.6416015625],
            [394.2544860839844, 462.77801513671875],
            [393.8092346191406, 473.9307861328125],
        ]
    )


def _load_line():
    return np.array(
        [
            [157.9123077392578, 408.2632141113281],
            [163.64076232910156, 407.1719055175781],
            [169.3692169189453, 406.08062744140625],
            [175.09765625, 404.98931884765625],
            [180.82611083984375, 403.8980407714844],
            [186.5545654296875, 402.8067321777344],
            [192.28302001953125, 401.7154235839844],
            [198.011474609375, 400.6241455078125],
            [203.73992919921875, 399.5328369140625],
            [209.46836853027344, 398.4415588378906],
            [215.1968231201172, 397.3502502441406],
            [220.92527770996094, 396.2589416503906],
            [226.6537322998047, 395.16766357421875],
            [232.38217163085938, 394.07635498046875],
            [238.11062622070312, 392.9850769042969],
            [243.83908081054688, 391.8937683105469],
            [249.56753540039062, 390.802490234375],
            [255.29598999023438, 389.711181640625],
            [261.0244445800781, 388.619873046875],
            [266.7528991699219, 387.5285949707031],
            [272.48133850097656, 386.4372863769531],
            [278.20977783203125, 385.34600830078125],
            [283.938232421875, 384.25469970703125],
            [289.66668701171875, 383.16339111328125],
            [295.3951416015625, 382.0721130371094],
            [301.12359619140625, 380.9808044433594],
            [306.85205078125, 379.8895263671875],
            [312.58050537109375, 378.7982177734375],
            [318.3089599609375, 377.7069091796875],
            [324.03741455078125, 376.6156311035156],
            [329.765869140625, 375.5243225097656],
            [335.49432373046875, 374.43304443359375],
            [341.2227783203125, 373.34173583984375],
            [346.95123291015625, 372.25042724609375],
            [352.6796569824219, 371.1591491699219],
            [358.4081115722656, 370.0678405761719],
            [364.1365661621094, 368.9765625],
            [369.864990234375, 367.88525390625],
            [375.5934753417969, 366.7939453125],
            [381.3218994140625, 365.7026672363281],
            [387.05035400390625, 364.6113586425781],
            [392.77880859375, 363.52008056640625],
            [398.50726318359375, 362.42877197265625],
            [404.2357177734375, 361.33746337890625],
            [409.96417236328125, 360.2461853027344],
            [415.692626953125, 359.1549072265625],
            [421.42108154296875, 358.0635986328125],
            [427.1495361328125, 356.9722900390625],
            [432.87799072265625, 355.8809814453125],
            [438.6064453125, 354.7897033691406],
            [444.33489990234375, 353.6983947753906],
            [450.0633544921875, 352.60711669921875],
            [455.791748046875, 351.51580810546875],
            [461.520263671875, 350.4245300292969],
            [467.24871826171875, 349.3332214355469],
            [472.9771728515625, 348.2419128417969],
            [478.70556640625, 347.150634765625],
            [484.43402099609375, 346.059326171875],
            [490.1624755859375, 344.9680480957031],
            [495.89093017578125, 343.8767395019531],
            [501.619384765625, 342.78546142578125],
            [507.34783935546875, 341.69415283203125],
            [513.0762939453125, 340.60284423828125],
            [518.8047485351562, 339.5115661621094],
            [524.533203125, 338.4202575683594],
            [530.2616577148438, 337.3289794921875],
            [535.9901123046875, 336.2376708984375],
            [541.7185668945312, 335.1463623046875],
            [547.447021484375, 334.0550842285156],
            [553.1754760742188, 332.9637756347656],
            [558.9039306640625, 331.87249755859375],
            [564.6323852539062, 330.78118896484375],
            [570.36083984375, 329.68988037109375],
            [576.0892944335938, 328.5986022949219],
            [581.8176879882812, 327.5072937011719],
            [587.5462036132812, 326.416015625],
            [593.274658203125, 325.32470703125],
            [599.0030517578125, 324.2333984375],
            [604.7315063476562, 323.1421203613281],
            [610.4599609375, 322.0508117675781],
            [616.1884155273438, 320.95953369140625],
            [621.9168701171875, 319.86822509765625],
            [627.6453247070312, 318.77691650390625],
            [633.373779296875, 317.6856384277344],
            [639.1022338867188, 316.5943603515625],
            [644.8306884765625, 315.5030517578125],
            [650.5591430664062, 314.4117431640625],
            [656.28759765625, 313.3204345703125],
            [662.0160522460938, 312.2291564941406],
            [667.7445068359375, 311.1378479003906],
            [673.472900390625, 310.04656982421875],
            [679.2013549804688, 308.95526123046875],
            [684.9298095703125, 307.8639831542969],
            [690.6582641601562, 306.7726745605469],
            [696.38671875, 305.6813659667969],
            [702.115234375, 304.590087890625],
            [707.8436889648438, 303.498779296875],
            [713.5720825195312, 302.4075012207031],
            [719.300537109375, 301.3161926269531],
            [725.0289916992188, 300.22491455078125],
        ]
    )


load_data("pigeon")
