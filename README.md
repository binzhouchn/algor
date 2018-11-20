# 数据结构与算法(python)

### chapter_2 二分法(Binary Search)

 - [二分查找](#二分查找)


### chapter_3 二叉树与分治法(Binary Tree & Divide Conquer)


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


[第一个错误的代码版本](https://www.lintcode.com/problem/first-bad-version/description)<br>
[在大数组中查找](https://www.lintcode.com/problem/search-in-a-big-sorted-array/description)<br>
[寻找旋转排序数组中的最小值](https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description)<br>

