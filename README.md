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

第一境界（二分位置 之 OOXX）

 - [二分查找](#二分查找)
 - [第一个错误的代码版本](#第一个错误的代码版本)
 - [在大数组中查找](#在大数组中查找)
 - [寻找旋转排序数组中的最小值](#寻找旋转排序数组中的最小值)
 - [搜索二维矩阵](#搜索二维矩阵)
 - [搜索二维矩阵 II](#搜索二维矩阵2)
 - [搜索区间](#搜索区间)
 - [目标出现总和](#目标出现总和)
 - [山脉序列中的最大值](#山脉序列中的最大值)

第二境界（二分位置 之 Half half）

 - [寻找峰值](#寻找峰值)
 - [搜索旋转排序数组star](#搜索旋转排序数组)
 
第三境界（二分答案）

 - [x的平方根](#x的平方根)
 - [木材加工](#木材加工)
 - [书籍复印](#书籍复印)
 - [两个整数相除](#两个整数相除)

**chapter_3 二叉树与分治法(Binary Tree & Divide Conquer)**

 - [二叉树的前序遍历](#二叉树的前序遍历)
 - [二叉树的中序遍历](#二叉树的中序遍历)
 - [二叉树的后序遍历](#二叉树的后序遍历)

破枪式（碰到二叉树的问题，就想想整棵树在该问题上的结果
和左右儿子在该问题上的结果之间的联系是什么）

 - [二叉树的最大深度](#二叉树的最大深度)
 - [二叉树的所有路径](#二叉树的所有路径)
 - [最小子树](#最小子树)
 - [平衡二叉树](#平衡二叉树)
 - [具有最大平均数的子树](#具有最大平均数的子树)
 - [将二叉树拆成链表](#将二叉树拆成链表)
 - [Lowest Common Ancestor of a Binary Tree](#lowest_common_ancestor_of_a_binary_tree)
 - [二叉树最长连续序列](#二叉树最长连续序列)
 - [二叉树的路径和](#二叉树的路径和)
 - [二叉树的路径和 II](#二叉树的路径和2)
 - [验证二叉查找树](#验证二叉查找树)
 - [二叉查找树迭代器](#二叉查找树迭代器)
 - [二叉树的层次遍历](#二叉树的层次遍历)
 

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
            # mid = ((end - start) >> 1) + start #注意python运算符和优先级
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

### 搜索二维矩阵2

[搜索二维矩阵 II](https://www.lintcode.com/problem/search-a-2d-matrix-ii/description)<br>
```python
class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix or len(matrix) == 0:
            return 0
        col = 0
        row = len(matrix) - 1
        res = 0
        while col < len(matrix[0]) and row > -1:
            if matrix[row][col] == target:
                res += 1
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return res
```

### 搜索区间

[搜索区间](https://www.lintcode.com/problem/search-for-a-range/description)
思路：先搜索target最左边的值，然后再把这个值赋给start重新搜索最右边的值<br>
```python
class Solution:
    """
    @param A : a list of integers
    @param target : an integer to be searched
    @return : a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        if not A or len(A) == 0:
            return [-1, -1]
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if A[mid] < target:
                start = mid
            else:
                end = mid
        
        if A[start] == target:
            leftBound = start
        elif A[end] == target:
            leftBound = end
        else:
            return [-1, -1]
        
        start, end = leftBound, len(A) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if A[mid] <= target:
                start = mid
            else:
                end = mid
        if A[end] == target:
            rightBound = end
        else:
            rightBound = start
        return [leftBound, rightBound]
```

### 山脉序列中的最大值

[山脉序列中的最大值](https://www.lintcode.com/problem/maximum-number-in-mountain-sequence/description)<br>
时间复杂度O(n)<br>
```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        if len(nums) == 1:
            return nums[0]
        i = 0
        j = 1
        while j < len(nums):
            if nums[i] > nums[j]:
                return nums[i]
            i += 1
            j += 1
        return -9999
```

时间复杂度O(logn)<br>
```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        if len(nums) == 1:
            return nums[0]
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if nums[mid] < nums[mid + 1]:
                start = mid
            else:
                end = mid
        if nums[start] > nums[end]:
            return nums[start]
        else:
            return nums[end]
```

### 寻找峰值

[寻找峰值](https://www.lintcode.com/problem/find-peak-element/description)<br>
```python
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        if not A or len(A) < 3:
            return -1
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if A[mid] > A[mid - 1] and A[mid] > A[mid + 1]:
                return mid
            elif A[mid] > A[mid - 1] and A[mid] < A[mid + 1]:
                start = mid
            else:
                end = mid
        return -1
```

### 搜索旋转排序数组

[搜索旋转排序数组](https://www.lintcode.com/problem/search-in-rotated-sorted-array/description)<br>
```python
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        if not A or len(A) == 0:
            return -1
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if A[mid] == target:
                return mid
            # 画图 看纸质ppt
            if A[start] < A[mid]:
                if A[start] <= target and target <= A[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if A[mid] <= target and target <= A[end]:
                    start = mid
                else:
                    end = mid
        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1
```

### x的平方根

[x的平方根](https://www.lintcode.com/problem/sqrtx/)<br>
```python
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):
        # write your code here
        if x < 1:
            return 0
        if x == 1:
            return 1
        start = 1
        end = x
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if mid * mid <= x and (mid+1)*(mid+1) >= x:
                return mid
            elif mid * mid > x:
                end = mid
            else:
                start = mid
        return 0
```

### 木材加工

[木材加工](https://www.lintcode.com/problem/wood-cut/description)<br>
```python
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        # write your code here
        if not L or len(L) == 0 or k < 1:
            return 0
        start = 1
        end = max(L)
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            r = sum([x//mid for x in L])
            if r == k:
                start = mid
            elif r > k:
                start = mid
            else:
                end = mid
        print(start)
        print(end)
        if sum([x//end for x in L]) >= k:
            return end
        if sum([x//start for x in L]) >= k:
            return start
        return 0
```

### 书籍复印

[书籍复印](https://www.lintcode.com/problem/copy-books/description)<br>
一句话描述题意：将数组切分为k个子数组，让数组和最大的最小<br>
```
令狐冲解法：
使用九章算法强化班中讲过的基于答案值域的二分法。
答案的范围在 max(pages)~sum(pages) 之间，每次二分到一个时间 time_limit 的时候，用贪心法从左到右扫描一下 pages，看看需要多少个人来完成抄袭。
如果这个值 <= k，那么意味着大家花的时间可能可以再少一些，如果 > k 则意味着人数不够，需要降低工作量。

时间复杂度 O(nlog(sum))O(nlog(sum)) 是该问题时间复杂度上的最优解法
```
```python
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        # write your code here
        if not pages or len(pages) == 0 or k < 1:
            return 0
        start = max(pages)
        end = sum(pages)
        if k == 1:
            return end
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if self.check(pages, mid) <= k:
                end = mid
            else:
                start = mid
        if self.check(pages, start) <= k:
            return start
        return end
        
    def check(self, pages, time_limit):
        count = 1
        cul_pages = 0
        for p in pages:
            if cul_pages + p > time_limit:
                count += 1
                cul_pages = 0
            cul_pages += p
        return count
```

### 两个整数相除

[两个整数相除](https://www.lintcode.com/problem/divide-two-integers/description)
```python
class Solution:
    """
    @param dividend: the dividend
    @param divisor: the divisor
    @return: the result
    """
    def divide(self, dividend, divisor):
        # write your code here
        INT_MAX = 2147483647
        if divisor == 0:
            return INT_MAX
        if dividend == 0:
            return 0
        neg = (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res, shift = 0, 31
        while shift >= 0:
            # 100 >= (9<<3)才会进到if里面
            if dividend >= (divisor << shift):
                dividend -= (divisor << shift)
                res += (1 << shift)
            shift -= 1
        if neg:
            res = -res
        if res > INT_MAX:
            return INT_MAX
        return res
```

--- 

### 二叉树的前序遍历

[二叉树的前序遍历](https://www.lintcode.com/problem/binary-tree-preorder-traversal/description)<br>
```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
# version1 Recursion(Divide & Conquer)
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        # 根左右
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

# version2 Recursion(Traverse)
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        res = []
        if root is None:
            return res
        self.helper(root, res)
        
        return res
    
    def helper(self, root, res):
        if root is None:
            return 
        res.append(root.val)
        self.helper(root.left, res)
        self.helper(root.right, res)

# version3 Non-Recursion
class Solution:
    """
    @param root: The root of binary tree.
    @return: Preorder in list which contains node values.
    """
    def preorderTraversal(self, root):
        if root is None:
            return []
        stack = [root]
        preorder = []
        while stack:
            node = stack.pop()
            preorder.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return preorder
```
### 二叉树的中序遍历

[二叉树的中序遍历](https://www.lintcode.com/problem/binary-tree-inorder-traversal/description)<br>
```python
# Divide & Conquer
class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        # 左根右
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

### 二叉树的后序遍历

[二叉树的后序遍历](https://www.lintcode.com/problem/binary-tree-postorder-traversal/description)<br>
```python
# version1 Divide & Conquer
class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

# version2 Traverse
class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        res = []
        if root is None:
            return res
        self.helper(root, res)
        return res
    def helper(self, root, res):
        if root is None:
            return
        self.helper(root.left, res)
        self.helper(root.right, res)
        res.append(root.val)
```

### 二叉树的最大深度

[二叉树的最大深度](https://www.lintcode.com/problem/maximum-depth-of-binary-tree/description)<br>
```python
# Divide & Conquer
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        # write your code here
        if root is None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
```

### 二叉树的所有路径

[二叉树的所有路径](https://www.lintcode.com/problem/binary-tree-paths/description)<br>
```python
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        # write your code here
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [str(root.val)]
        left = self.binaryTreePaths(root.left)
        right = self.binaryTreePaths(root.right)
        
        paths = []
        for p in left:
            paths.append(str(root.val) + '->' + p)
        for p in right:
            paths.append(str(root.val) + '->' + p)
        return paths
```

### 最小子树

[最小子树](https://www.lintcode.com/problem/minimum-subtree/description)<br>
```python
# Traverse + Divide Conquer
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        if root is None:
            return None
        self.min_weight = sys.maxsize
        self.res = None
        self.helper(root)
        return self.res
    def helper(self, root):
        if root is None:
            return 0
        left_weight = self.helper(root.left)
        right_weight = self.helper(root.right)
        
        tmp_sum = left_weight + right_weight + root.val
        if tmp_sum < self.min_weight:
            self.min_weight = tmp_sum
            self.res = root
        return tmp_sum
```

### 平衡二叉树

[平衡二叉树](https://www.lintcode.com/problem/balanced-binary-tree/description)<br>
```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        if root is None:
            return True
        isbal, _ = self.helper(root)
        return isbal
        
    def helper(self, root):
        if root is None:
            return True, 0
        isbal_left, height_left = self.helper(root.left)
        if not isbal_left:
            return False, 0
        isbal_right, height_right = self.helper(root.right)
        if not isbal_right:
            return False, 0
        return abs(height_left - height_right) <= 1, max(height_left, height_right) + 1
```

### 具有最大平均数的子树

[具有最大平均数的子树](https://www.lintcode.com/problem/subtree-with-maximum-average/description)<br>
```python
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    def findSubtree2(self, root):
        # write your code here
        if root is None:
            return None
        self.max_avg = -sys.maxsize
        self.res = None
        self.helper(root)
        return self.res
        
    def helper(self, root):
        if root is None:
            return 0, 0
        left_sum, left_count = self.helper(root.left)
        right_sum, right_count = self.helper(root.right)
        tmp_avg = (left_sum + right_sum + root.val) / (left_count + right_count + 1)
        if tmp_avg > self.max_avg:
            self.max_avg = tmp_avg
            self.res = root
        return left_sum + right_sum + root.val, left_count + right_count + 1
```

### 将二叉树拆成链表

[将二叉树拆成链表](https://www.lintcode.com/problem/flatten-binary-tree-to-linked-list/description)<br>
```python
# 我的思路：用前序遍历一遍node存在list中，然后依次连接left=None,right=next
# 需要使用额外的空间耗费(挑战)
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    res = []
    def flatten(self, root):
        # write your code here
        if root is None:
            return
        self.helper(root)
        for i in range(len(self.res)-1):
            self.res[i].left = None
            self.res[i].right = self.res[i+1]
        root = self.res[0]
        
    def helper(self, root):
        if root is None:
            return
        self.res.append(root)
        self.helper(root.left)
        self.helper(root.right)

# 令狐冲老师代码（不使用额外的空间耗费）
# Divide & Conquer
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        if root is None:
            return
        self.helper(root)
    def helper(self, root):
        if root is None:
            return None
        left = self.helper(root.left)
        right = self.helper(root.right)
        if left is not None:
            left.right = root.right
            root.right = root.left
            root.left = None
        if right is not None:
            return right
        if left is not None:
            return left
        return root
```

### Lowest_Common_Ancestor_of_a_Binary_Tree

[Lowest Common Ancestor of a Binary Tree](https://www.lintcode.com/problem/lowest-common-ancestor-of-a-binary-tree/description)<br>
```python
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        if root is None:
            return None
        if root == A or root == B:
            return root
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        if left and right:
            return root
        if left:
            return left
        if right:
            return right
        return None
```

### 二叉树最长连续序列

[二叉树最长连续序列](https://www.lintcode.com/problem/binary-tree-longest-consecutive-sequence/description)<br>
```python
class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code here
        return self.helper(root, None, 0)
    def helper(self, root, parent, lengthwithoutroot):
        if root is None:
            return 0
        # 加上root以后的看能形成的长度
        length = lengthwithoutroot + 1 if (parent is not None and parent.val + 1 == root.val) else 1
        left = self.helper(root.left, root, length)
        right = self.helper(root.right, root, length)
        return max(length, max(left, right))
```

### 二叉树的路径和

[二叉树的路径和](https://www.lintcode.com/problem/binary-tree-path-sum/description)<br>
```python
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    res = []
    def binaryTreePathSum(self, root, target):
        # write your code here
        if root is None:
            return []
        path = []
        self.helper(root, path, target)
        return self.res
    def helper(self, root, path, target):
        if root is None:
            return []
        path.append(root.val)
        if root.left is None and root.right is None:
            if sum(path) == target:
                self.res.append(path.copy())
        self.helper(root.left, path.copy(), target)
        self.helper(root.right, path.copy(), target)
```

### 二叉树的路径和2

[二叉树的路径和2](https://www.lintcode.com/problem/binary-tree-path-sum-ii/description)<br>
```python
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    res = []
    def binaryTreePathSum2(self, root, target):
        # write your code here
        if root is None:
            return []
        self.helper(root, target, [], 0)
        return self.res
    def helper(self, root, target, path, l):
        if root is None:
            return
        path.append(root.val)
        tmp = target
        for i in range(l, -1, -1):
            tmp -= path[i]
            if tmp == 0:
                self.res.append(path[i:].copy())
        self.helper(root.left, target, path, l+1)
        self.helper(root.right, target, path, l+1)
        path.pop()
```

### 验证二叉查找树

[验证二叉查找树](https://www.lintcode.com/problem/validate-binary-search-tree/description)<br>
```python
class resulttype:
    def __init__(self, isbst, max_val, min_val):
        self.isbst = isbst
        self.max_val = max_val
        self.min_val = min_val

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        res = self.helper(root)
        return res.isbst
    def helper(self, root):
        if root is None:
            return resulttype(True, -sys.maxsize, sys.maxsize)
        left = self.helper(root.left)
        right = self.helper(root.right)
        if (not left.isbst or not right.isbst):
            return resulttype(False, 0, 0)
        if (left is not None and left.max_val >= root.val) or (right is not None and right.min_val <= root.val):
            return resulttype(False, 0, 0)
        return resulttype(True, max(root.val, right.max_val), min(root.val, left.min_val))
```

### 二叉查找树迭代器

[二叉查找树迭代器](https://www.lintcode.com/problem/binary-search-tree-iterator/description)<br>
```python
class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        return len(self.stack) > 0

    """
    @return: return next node
    """
    def next(self):
        node = self.stack[-1]
        if node.right:
            n = node.right
            while n:
                self.stack.append(n)
                n = n.left
        else:
            n = self.stack.pop()
            while self.stack and self.stack[-1].right == n:
                n = self.stack.pop()
        return node
```

### 二叉树的层次遍历

[二叉树的层次遍历](https://www.lintcode.com/problem/binary-tree-level-order-traversal/description)<br>
```python
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
```

### 