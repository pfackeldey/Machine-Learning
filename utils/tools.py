# -*- coding: utf-8 -*-

import re
import os


def flattenList(listOfLists):
    """
    flatten 2D list
    return [1, 2, 3, 4, ...] for input [[1, 2], [3, 4, ...], ...]
    """
    return [item for subList in listOfLists for item in subList]


def matchingItem(regexItems, string):
    """
    find/return item in string list that matches the given regex
    returns None in case no matching item is found
    """
    for item in regexItems:
        if re.search(item, string) != None:
            return item
    return None


def get_folder_size(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += get_folder_size(itempath)
    return total_size
