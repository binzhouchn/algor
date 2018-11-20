---
title:  "编程-LintCode"
layout: post
categories: python 编程-lintcode
tags:  python
author: 周彬
---

* content
{:toc}

# 数据结构与算法(python)


**chapter_2 二分法(Binary Search)**

 - [二分查找](#二分查找)
 - [第一个错误的代码版本](#第一个错误的代码版本)
 - [在大数组中查找](#在大数组中查找)
 - [寻找旋转排序数组中的最小值](#寻找旋转排序数组中的最小值)
 - [搜索二维矩阵](#搜索二维矩阵)


**chapter_3 二叉树与分治法(Binary Tree & Divide Conquer)**

待补充

**chapter_4 宽度优先搜索(Breadth First Search)**

待补充

**chapter_5 深度优先搜索(Depth First Search)**

待补充

**chapter_6 链表与数组(Linked List & Array)**

待补充

**chapter_7 两根指针(Two Pointers)**

待补充

**chapter_8 数据结构(Data Structure)**

待补充

**chapter_9 动态规划(Dynamic Programming)**

待补充

---

### 二分查找

[二分查找问题描述](http://www.lintcode.com/problem/first-position-of-target/)<br>
```python
class Solution:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """
    def binarySearch(self, nums, target):
        # write your code here
        if not nums or len(nums) == 0:
            return -1
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            # 细节，不写成(start + end) / 2：为了防止溢出2^32
            mid = int((end - start) / 2 + start)
            # mid = (end - start) >> 1 + start
            if target == nums[mid]:
                end = mid
            elif target < nums[mid]:
                end = mid
            else:
                start = mid
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1
```

### 第一个错误的代码版本

[第一个错误的代码版本](https://www.lintcode.com/problem/first-bad-version/description)<br>
```python
#class SVNRepo:
#    @classmethod
#    def isBadVersion(cls, id)
#        # Run unit tests to check whether verison `id` is a bad version
#        # return true if unit tests passed else false.
# You can use SVNRepo.isBadVersion(10) to check whether version 10 is a
# bad version.
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        if n < 1:
            return -1
        if n == 1 and SVNRepo.isBadVersion(1):
            return 1
        start = 1
        end = n
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid
        if SVNRepo.isBadVersion(start):
            return start
        if SVNRepo.isBadVersion(end):
            return end
        return -1
```

### 在大数组中查找

[在大数组中查找](https://www.lintcode.com/problem/search-in-a-big-sorted-array/description)<br>
```python
class Solution:
    """
    @param: reader: An instance of ArrayReader.
    @param: target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        if not reader:
            return -1
        start = 0
        end = 1
        while reader.get(end) < target:
            end *= 2
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if reader.get(mid) == target:
                end = mid
            elif reader.get(mid) > target:
                end = mid
            else:
                start = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1
```

### 寻找旋转排序数组中的最小值

[寻找旋转排序数组中的最小值](https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description)<br>
```python
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        if nums[start] < nums[end]:
            return nums[start]
        return nums[end]
```

### 搜索二维矩阵

[搜索二维矩阵](https://www.lintcode.com/problem/search-a-2d-matrix/description)<br>
```python
class Solution:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix or len(matrix) == 0:
            return False
        col = 0
        row = len(matrix) - 1
        while col < len(matrix[0]) and row > -1:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return False
```
