# 数据结构与算法(python)


**chapter_2 二分法(Binary Search)**

 - [二分查找](#二分查找)
 - [第一个错误的代码版本](#第一个错误的代码版本)
 - [在大数组中查找](#在大数组中查找)
 - [寻找旋转排序数组中的最小值](#寻找旋转排序数组中的最小值)


**chapter_3 二叉树与分治法(Binary Tree & Divide Conquer)**

待补充

**chapter_4 xxx**

**chapter_5 xxx**

**chapter_6 xxx**

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
            mid = int((end - start) / 2 + start)
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

