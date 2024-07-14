def bin_search_schedule(schedule,
                        iter_idx: int = 0):
    if len(schedule) == 1:
        return schedule[0][1]
    else:
        mid = len(schedule) // 2

        if schedule[mid][0] <= iter_idx:
            return bin_search_schedule(schedule[mid:], iter_idx)
        else:
            return bin_search_schedule(schedule[:mid], iter_idx)
