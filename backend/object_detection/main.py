from od_utils_delfi import *

# # x = process_folder("/home/mortirreke/Desktop/assets/")
# #
# x = load_data()
#
# # image = cv2.imread("/home/mortirreke/Desktop/assets/nicolas-poussin_holy-family-1650.jpg")
# image = cv2.imread("../prova/comfort/16050.jpg")
# objs = get_objects(image)
#
# print(objs)
# # result2 = find_matchesVec(objs, x, False)
#
# # print(result2[:10])
#






# image = cv2.imread("../prova/comfort/17942.jpg")
# objs = get_objects_deets(image)
#
# print(objs)

get_folder_objects_json("../prova/comfort", "output.json")
