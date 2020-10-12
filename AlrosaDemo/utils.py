import pickle
from alrosademo.KeyFilter import filter_data, work_with_obs_indexes



# def getframedtonohanda(obs, fpsvalue, frames_count):
#     frames_numbers = work_with_obs_indexes(obs)
#     frames_numbers = sorted(list(set(frames_numbers)))
#     frames_numbers = [0] + frames_numbers + [frames_count]
#     secdur = 3
#     selected = []
#     for i in range(1,len(frames_numbers)):
#         if frames_numbers[i]-frames_numbers[i-1] > fpsvalue*secdur:
#             print(frames_numbers[i],frames_numbers[i-1])
#             selected += [j for j in range(frames_numbers[i-1]+fpsvalue*secdur, frames_numbers[i])]
#     return selected


def getframedtonohanda(obs, fpsvalue, frames_count):
    frames_numbers = work_with_obs_indexes(obs)
    frames_numbers = sorted(list(set(frames_numbers)))



    frames_count = int(frames_count)
    frames_numbers = [0] + frames_numbers + [frames_count]
    # print('frames_numbers', frames_numbers)
    secdur = 3
    fpsvalue=int(fpsvalue)

    selected = []
    for i in range(1,len(frames_numbers)):
        if frames_numbers[i]-frames_numbers[i-1] > fpsvalue*secdur:
            # print(frames_numbers[i],frames_numbers[i-1])

            # if frames_numbers[i]-frames_numbers[i-1] < 10:
            #     continue

            li = [j for j in range(frames_numbers[i-1]+fpsvalue*secdur, frames_numbers[i])]
            # if len(li)<10:
            #     continue
            li = li[::2]
            selected+=li
    if frames_numbers[1] >= fpsvalue // 2:
        selected += [i for i in range(frames_numbers[1])][::2]
    # print(selected)
    return selected