#### **Unit 6: Advanced Mathematics for ML**  

# Topic 1: Advanced Linear Algebra

Students will be assessed on their ability to:

**6.1.1 Understand singular value decomposition (SVD)**
- Define SVD and its mathematical representation
- Understand the relationship between SVD and eigenvalue decomposition
- Identify the properties of singular values and singular vectors
- Apply SVD to decompose matrices into constituent components

**Guidance:** Students should understand SVD as a fundamental matrix factorization that decomposes any matrix into three matrices (U, Σ, Vᵀ). They should recognize that SVD exists for all matrices, unlike eigenvalue decomposition which requires square matrices. The geometric interpretation of SVD as rotation, scaling, and rotation should be emphasized. Applications to dimensionality reduction and data compression should be referenced.

**6.1.2 Calculate singular values and singular vectors**
- Compute singular values of a given matrix
- Determine the corresponding left and right singular vectors
- Verify the SVD decomposition by matrix multiplication
- Interpret the meaning of singular values in terms of matrix properties

**Guidance:** Students should be able to compute singular values by finding the eigenvalues of AᵀA. They should understand that singular vectors are the eigenvectors of AAᵀ (left singular vectors) and AᵀA (right singular vectors). Students should verify their calculations by reconstructing the original matrix from its SVD components. The relationship between singular values and concepts like matrix rank and condition number should be explored.

**6.1.3 Apply truncated SVD for dimensionality reduction**
- Implement truncated SVD to reduce dimensionality of data
- Determine the optimal number of singular values to retain
- Evaluate the reconstruction error for different truncation levels
- Apply truncated SVD to practical datasets

**Guidance:** Students should understand how truncated SVD works by keeping only the k largest singular values and their corresponding vectors. They should be able to determine the appropriate number of components to retain based on explained variance or reconstruction error. The trade-off between dimensionality reduction and information loss should be analyzed. Applications to image compression, document clustering, and feature extraction should be demonstrated.

**6.1.4 Understand eigenvalue decomposition**
- Define eigenvalue decomposition and its mathematical representation
- Understand the conditions under which eigenvalue decomposition exists
- Identify the relationship between eigenvalues and matrix properties
- Apply eigenvalue decomposition to diagonalizable matrices

**Guidance:** Students should understand eigenvalue decomposition as expressing a matrix in terms of its eigenvalues and eigenvectors. They should recognize that not all matrices are diagonalizable and understand the conditions for diagonalizability. The connection between eigenvalues and matrix properties such as determinant, trace, and invertibility should be emphasized. Applications to systems of linear equations and dynamical systems should be referenced.

**6.1.5 Calculate eigenvalues and eigenvectors**
- Compute eigenvalues of square matrices using characteristic equations
- Determine the corresponding eigenvectors for each eigenvalue
- Verify eigenvalue-eigenvector pairs through matrix multiplication
- Handle cases with repeated eigenvalues and complex eigenvalues

**Guidance:** Students should be able to find eigenvalues by solving the characteristic equation det(A - λI) = 0. They should understand how to find eigenvectors by solving (A - λI)v = 0 for each eigenvalue. Students should verify their solutions by checking if Av = λv. Special cases including repeated eigenvalues and complex eigenvalues should be addressed, with emphasis on the geometric multiplicity versus algebraic multiplicity.

**6.1.6 Analyze matrix diagonalizability**
- Determine if a matrix is diagonalizable
- Find the diagonal matrix and transformation matrix for diagonalizable matrices
- Understand the relationship between diagonalizability and eigenvectors
- Handle non-diagonalizable matrices using Jordan form

**Guidance:** Students should understand that a matrix is diagonalizable if and only if it has n linearly independent eigenvectors, where n is the dimension of the matrix. They should be able to construct the diagonal matrix D and the transformation matrix P such that A = PDP⁻¹. The concept of algebraic versus geometric multiplicity and its impact on diagonalizability should be explored. For non-diagonalizable matrices, students should be introduced to Jordan canonical form as an alternative representation.

**6.1.7 Understand LU decomposition**
- Define LU decomposition and its mathematical representation
- Apply LU decomposition to solve systems of linear equations
- Implement LU decomposition with and without pivoting
- Analyze the computational complexity of LU decomposition

**Guidance:** Students should understand LU decomposition as factorizing a matrix into lower (L) and upper (U) triangular matrices. They should be able to use LU decomposition to efficiently solve multiple systems of equations with the same coefficient matrix. The importance of pivoting for numerical stability should be emphasized, with students able to implement LU decomposition with partial pivoting. The computational complexity of O(n³) for LU decomposition should be compared to other methods.

**6.1.8 Apply QR decomposition**
- Define QR decomposition using Gram-Schmidt orthogonalization
- Apply QR decomposition to solve least squares problems
- Implement QR decomposition using Householder transformations
- Understand the relationship between QR decomposition and orthogonal matrices

**Guidance:** Students should understand QR decomposition as factorizing a matrix into an orthogonal matrix Q and an upper triangular matrix R. They should be able to apply Gram-Schmidt orthogonalization to compute QR decomposition and recognize its numerical instability. The more stable Householder transformation method should be introduced. Applications to solving least squares problems and eigenvalue computation should be explored. The properties of orthogonal matrices and their computational advantages should be emphasized.

**6.1.9 Understand Cholesky decomposition**
- Define Cholesky decomposition for symmetric positive definite matrices
- Implement Cholesky decomposition algorithm
- Apply Cholesky decomposition to solve linear systems
- Analyze the computational advantages of Cholesky decomposition

**Guidance:** Students should understand Cholesky decomposition as A = LLᵀ, where L is a lower triangular matrix, and recognize that it applies only to symmetric positive definite matrices. They should be able to implement the Cholesky algorithm and verify positive definiteness through the decomposition process. The computational advantages of Cholesky decomposition (half the operations of LU decomposition) should be analyzed. Applications to optimization problems and Monte Carlo simulations should be referenced.

**6.1.10 Apply matrix decompositions to principal component analysis (PCA)**
- Understand the mathematical foundation of PCA through eigenvalue decomposition
- Implement PCA using covariance matrix eigenvalue decomposition
- Apply PCA for dimensionality reduction and feature extraction
- Interpret principal components and explained variance

**Guidance:** Students should understand PCA as finding the orthogonal directions of maximum variance in data through eigenvalue decomposition of the covariance matrix. They should be able to implement PCA by computing the covariance matrix, finding its eigenvalues and eigenvectors, and projecting data onto the principal components. The concept of explained variance and its relationship to eigenvalues should be explored. Students should be able to determine the optimal number of principal components to retain based on explained variance criteria.

**6.1.11 Apply matrix decompositions to recommendation systems**
- Implement collaborative filtering using matrix factorization
- Understand the connection between SVD and recommendation algorithms
- Apply matrix completion techniques for sparse datasets
- Evaluate recommendation quality using appropriate metrics

**Guidance:** Students should understand how recommendation systems can be formulated as matrix completion problems, where the goal is to predict missing entries in a user-item rating matrix. They should be able to implement basic collaborative filtering using matrix factorization techniques like SVD. The challenges of applying SVD to sparse matrices with missing values should be addressed, with solutions like imputation or specialized algorithms introduced. Evaluation metrics for recommendation systems (RMSE, precision, recall) should be applied to assess recommendation quality.

**6.1.12 Apply matrix decompositions to data compression**
- Use SVD for image compression
- Analyze the trade-off between compression ratio and reconstruction quality
- Implement compression using truncated matrix representations
- Compare different matrix decomposition methods for compression efficiency

**Guidance:** Students should understand how SVD enables data compression by keeping only the most significant singular values and vectors. They should be able to implement image compression by representing an image as a matrix, applying SVD, truncating to keep only k singular values, and reconstructing the compressed image. The relationship between the number of retained singular values and compression ratio should be analyzed. Students should compare the compression efficiency of different matrix decomposition methods and evaluate the quality using metrics like PSNR or SSIM.



# Topic 2: Eigenvalues, Eigenvectors, and Spectral Theory

Students will be assessed on their ability to:

**6.2.1 Understand characteristic polynomials**
- Define the characteristic polynomial of a matrix
- Calculate characteristic polynomials for 2×2 and 3×3 matrices
- Relate the coefficients of characteristic polynomials to matrix invariants
- Apply the Cayley-Hamilton theorem to matrix functions

**Guidance:** Students should understand the characteristic polynomial as det(λI - A) = 0. They should be able to compute characteristic polynomials for small matrices by hand and recognize that the coefficients relate to trace, determinant, and other matrix invariants. The Cayley-Hamilton theorem (every matrix satisfies its own characteristic equation) should be understood, with applications to computing matrix powers and functions. Students should recognize that finding roots of characteristic polynomials is one method to determine eigenvalues.

**6.2.2 Analyze algebraic and geometric multiplicity**
- Define algebraic multiplicity of an eigenvalue
- Define geometric multiplicity of an eigenvalue
- Determine the algebraic and geometric multiplicity for given matrices
- Understand the relationship between algebraic and geometric multiplicity

**Guidance:** Students should understand algebraic multiplicity as the multiplicity of an eigenvalue as a root of the characteristic polynomial. They should understand geometric multiplicity as the dimension of the eigenspace corresponding to an eigenvalue (nullity of A - λI). Students should be able to compute both multiplicities for given matrices and recognize that geometric multiplicity is always less than or equal to algebraic multiplicity. The implications of these multiplicities for diagonalizability should be explored.

**6.2.3 Understand invariant subspaces**
- Define invariant subspaces under linear transformations
- Identify invariant subspaces associated with eigenvalues
- Analyze the direct sum decomposition of vector spaces
- Apply invariant subspace concepts to block diagonalization

**Guidance:** Students should understand an invariant subspace as a subspace W where T(W) ⊆ W for a linear transformation T. They should recognize that eigenspaces are examples of invariant subspaces. The concept of generalized eigenspaces and their role in the Jordan canonical form should be introduced. Students should understand how invariant subspaces enable block diagonalization of matrices and the computational advantages this provides. Applications to solving systems of linear differential equations should be referenced.

**6.2.4 Compute matrix functions using eigenvalues**
- Define matrix functions using eigenvalue decompositions
- Calculate matrix exponentials using eigenvalues and eigenvectors
- Apply matrix functions to solve systems of differential equations
- Understand the convergence of matrix power series

**Guidance:** Students should understand how to define functions of matrices using eigenvalue decomposition: if A = PDP⁻¹, then f(A) = Pf(D)P⁻¹. They should be able to compute matrix exponentials e^At, which are essential for solving systems of linear differential equations. The convergence of matrix power series should be analyzed in terms of the spectral radius. Applications to Markov chains, dynamical systems, and differential equations should be explored.

**6.2.5 Define and analyze the spectrum of a matrix**
- Define the spectrum of a matrix as the set of eigenvalues
- Understand the properties of the spectrum for different matrix classes
- Relate spectral properties to matrix characteristics
- Analyze the spectrum of special matrix types (symmetric, orthogonal, etc.)

**Guidance:** Students should understand the spectrum σ(A) as the set of all eigenvalues of A. They should recognize how the spectrum differs for various matrix classes: symmetric matrices have real spectra, orthogonal matrices have spectra on the unit circle, etc. The relationship between spectral properties and matrix characteristics like definiteness, invertibility, and stability should be explored. Students should be able to analyze the spectrum of special matrix types and draw conclusions about their properties.

**6.2.6 Understand spectral radius and its implications**
- Define spectral radius as the maximum absolute eigenvalue
- Calculate the spectral radius for given matrices
- Apply spectral radius to analyze matrix convergence
- Use spectral radius to determine stability of dynamical systems

**Guidance:** Students should understand spectral radius ρ(A) as max{|λ| : λ ∈ σ(A)}. They should be able to compute spectral radius for given matrices and recognize its importance in determining the convergence of matrix powers and iterative methods. The relationship between spectral radius and matrix norms should be explored, particularly that ρ(A) ≤ ||A|| for any matrix norm. Applications to stability analysis of linear dynamical systems and convergence of iterative algorithms should be emphasized.

**6.2.7 Apply spectral decomposition to normal matrices**
- Define normal matrices and their properties
- Understand the spectral theorem for normal matrices
- Implement spectral decomposition for symmetric and Hermitian matrices
- Apply spectral decomposition to quadratic forms and optimization

**Guidance:** Students should understand normal matrices as those that commute with their conjugate transpose (AA* = A*A). They should recognize the spectral theorem: a matrix is normal if and only if it is unitarily diagonalizable. Special cases including symmetric, Hermitian, skew-symmetric, and unitary matrices should be analyzed. Students should be able to perform spectral decomposition and apply it to analyzing quadratic forms, determining definiteness, and solving optimization problems.

**6.2.8 Analyze matrix convergence using spectral theory**
- Define convergence of matrix sequences and series
- Apply spectral radius to determine convergence of matrix powers
- Understand the Neumann series and its convergence conditions
- Analyze convergence of iterative methods using spectral theory

**Guidance:** Students should understand convergence of matrix sequences in terms of entry-wise convergence. They should recognize that matrix powers A^k converge to zero if and only if ρ(A) < 1. The Neumann series (I - A)⁻¹ = ΣA^k and its convergence when ρ(A) < 1 should be analyzed. Applications to convergence analysis of iterative methods like Jacobi, Gauss-Seidel, and power iteration should be explored. The relationship between convergence rate and spectral radius should be emphasized.

**6.2.9 Define and solve generalized eigenvalue problems**
- Define generalized eigenvalues and eigenvectors
- Formulate the generalized eigenvalue problem Av = λBv
- Understand special cases of generalized eigenvalue problems
- Solve generalized eigenvalue problems for specific matrix pairs

**Guidance:** Students should understand generalized eigenvalue problems as finding λ and v such that Av = λBv, where A and B are n×n matrices. They should recognize that when B is invertible, this reduces to a standard eigenvalue problem B⁻¹Av = λv. Special cases including definite generalized eigenvalue problems (where A and B are symmetric and B is positive definite) should be analyzed. Students should be able to solve small generalized eigenvalue problems by hand and understand numerical approaches for larger problems.

**6.2.10 Apply generalized eigenvalue problems to practical applications**
- Use generalized eigenvalue problems in vibration analysis
- Apply generalized eigenvalue decomposition to signal processing
- Understand the connection to quadratic forms and constrained optimization
- Implement algorithms for solving generalized eigenvalue problems

**Guidance:** Students should understand how generalized eigenvalue problems arise in mechanical vibration analysis (solving (K - λM)v = 0, where K is stiffness matrix and M is mass matrix). Applications to signal processing, particularly in MUSIC and ESPRIT algorithms for frequency estimation, should be explored. The connection to generalized Rayleigh quotients and constrained optimization problems should be analyzed. Students should be able to implement basic algorithms for solving generalized eigenvalue problems, such as the QZ algorithm.

**6.2.11 Apply spectral clustering for data grouping**
- Understand the mathematical foundation of spectral clustering
- Construct similarity graphs and graph Laplacians
- Apply eigenvalue decomposition of graph Laplacians for clustering
- Evaluate the quality of spectral clustering results

**Guidance:** Students should understand spectral clustering as using the eigenvalues and eigenvectors of similarity matrices to perform dimensionality reduction before clustering. They should be able to construct different types of similarity graphs (ε-neighborhood, k-nearest neighbor, fully connected) and compute graph Laplacians (unnormalized, normalized symmetric, normalized random walk). The connection between graph partitioning and eigenvectors of graph Laplacians should be explored. Students should be able to implement spectral clustering algorithms and evaluate their performance using metrics like silhouette score or normalized mutual information.

**6.2.12 Apply spectral embedding for dimensionality reduction**
- Define spectral embedding as a nonlinear dimensionality reduction technique
- Implement Laplacian Eigenmaps for manifold learning
- Apply spectral embedding to visualize high-dimensional data
- Compare spectral embedding with other dimensionality reduction methods

**Guidance:** Students should understand spectral embedding as mapping data points to a lower-dimensional space using eigenvectors of graph Laplacians. They should be able to implement Laplacian Eigenmaps, which preserve local geometric properties of data. The algorithmic steps (constructing adjacency graph, computing weights, computing graph Laplacian, computing eigenvalue decomposition, embedding using eigenvectors) should be mastered. Applications to visualizing high-dimensional data in 2D or 3D should be explored. Students should compare spectral embedding with linear methods like PCA and nonlinear methods like t-SNE.

**6.2.13 Apply spectral graph theory to network analysis**
- Understand the spectrum of graph Laplacians and its properties
- Relate eigenvalues of graph Laplacians to graph properties
- Apply spectral graph theory to community detection
- Use spectral methods for graph partitioning and visualization

**Guidance:** Students should understand the spectrum of graph Laplacians and its connection to graph properties. They should recognize that the second smallest eigenvalue (Fiedler value) of the normalized Laplacian relates to graph connectivity, and its corresponding eigenvector (Fiedler vector) can be used for graph partitioning. Applications to community detection in social networks, image segmentation, and graph visualization should be explored. The relationship between eigenvalue multiplicity and graph symmetries should be analyzed.

**6.2.14 Apply spectral methods to semi-supervised learning**
- Understand the mathematical foundation of spectral methods in semi-supervised learning
- Implement label propagation algorithms using graph Laplacians
- Apply spectral regularization to semi-supervised classification
- Evaluate the performance of spectral semi-supervised learning methods

**Guidance:** Students should understand how spectral methods can be used in semi-supervised learning by leveraging both labeled and unlabeled data through graph-based approaches. They should be able to implement label propagation algorithms that use graph Laplacians to spread labels from labeled to unlabeled instances. The concept of spectral regularization and its application to semi-supervised classification should be explored. Students should be able to evaluate the performance of spectral semi-supervised learning methods and compare them with other approaches.



# Topic 3: Matrix Calculus for Machine Learning

Students will be assessed on their ability to:

**6.3.1 Define and compute derivatives of scalar functions with respect to vectors**
- Define the gradient of a scalar function with respect to a vector
- Compute partial derivatives of multivariate scalar functions
- Apply the chain rule for scalar functions of vector variables
- Calculate gradients of common scalar functions used in machine learning

**Guidance:** Students should understand the gradient as a vector of partial derivatives of a scalar function with respect to each element of a vector variable. They should be able to compute gradients for functions like linear forms (aᵀx), quadratic forms (xᵀAx), and norms (||x||₂). The chain rule for scalar functions of vector variables should be mastered, including cases where intermediate variables are also vectors. Students should practice computing gradients for common machine learning loss functions like mean squared error and cross-entropy.

**6.3.2 Define and compute derivatives of vector functions with respect to vectors**
- Define the Jacobian matrix for vector-valued functions
- Compute Jacobian matrices for simple vector transformations
- Apply the chain rule for compositions of vector functions
- Calculate derivatives of common vector operations in machine learning

**Guidance:** Students should understand the Jacobian matrix as containing all first-order partial derivatives of a vector-valued function. They should be able to compute Jacobians for linear transformations (Ax), element-wise functions (σ(x) where σ is applied element-wise), and other common vector operations. The chain rule for vector functions should be mastered, recognizing that matrix multiplication order matters in the composition. Students should practice computing Jacobians for operations commonly found in neural networks, such as activation functions and layer transformations.

**6.3.3 Define and compute derivatives of matrix functions**
- Define derivatives of scalar functions with respect to matrices
- Compute derivatives of common matrix functions (trace, determinant)
- Apply matrix calculus rules for matrix products and inverses
- Calculate gradients of matrix-variate functions used in machine learning

**Guidance:** Students should understand how to extend derivatives to matrix variables, recognizing that the derivative of a scalar function with respect to a matrix is another matrix of the same dimensions. They should be able to compute derivatives of trace (dtr(AX)/dX = Aᵀ) and determinant (d|X|/dX = |X|X⁻ᵀ) operations. Rules for derivatives of matrix products, inverses, and other matrix operations should be mastered. Students should practice computing gradients for matrix-variate functions common in machine learning, such as those appearing in matrix factorization and covariance matrix estimation.

**6.3.4 Apply layout conventions in matrix calculus**
- Understand numerator layout (Jacobian formulation) convention
- Understand denominator layout (Hessian formulation) convention
- Identify and convert between different layout conventions
- Apply consistent layout conventions in machine learning derivations

**Guidance:** Students should understand the two main layout conventions in matrix calculus: numerator layout (where derivatives are arranged with numerator dimensions changing fastest) and denominator layout (where denominator dimensions change fastest). They should recognize how these conventions affect the dimensions and arrangement of derivative matrices. Students should be able to identify which convention is being used in a given context and convert between conventions when necessary. The importance of maintaining consistent layout conventions throughout machine learning derivations should be emphasized.

**6.3.5 Define and apply matrix differentials**
- Define the matrix differential and its properties
- Compute differentials of matrix expressions
- Apply rules for matrix differentials (product rule, chain rule)
- Use differentials to derive matrix derivatives

**Guidance:** Students should understand the matrix differential as a first-order approximation to the change in a matrix function. They should be able to compute differentials for common matrix expressions and apply rules like the product rule (d(AB) = (dA)B + A(dB)) and chain rule. Students should learn how to use matrix differentials to derive derivatives by identifying the coefficient matrices in the differential expression. The advantage of using differentials for deriving complex matrix derivatives should be emphasized, particularly for functions involving matrix inverses and other nonlinear operations.

**6.3.6 Apply matrix differentials to derive derivatives of complex matrix expressions**
- Compute differentials of matrix inverses (d(A⁻¹) = -A⁻¹(dA)A⁻¹)
- Derive differentials of matrix traces and determinants
- Apply differentials to functions of matrix products and compositions
- Use differentials to derive gradients for optimization problems

**Guidance:** Students should master the differential of matrix inverse and understand how to derive it using the identity A⁻¹A = I. They should be able to compute differentials of traces of matrix expressions (d tr(AX) = tr(A dX)) and determinants (d|X| = |X| tr(X⁻¹ dX)). Students should practice applying differentials to more complex expressions involving matrix products, compositions, and element-wise operations. The application of these techniques to derive gradients for optimization problems in machine learning should be emphasized.

**6.3.7 Implement gradient descent for matrix-valued parameters**
- Formulate gradient descent updates for matrix parameters
- Apply appropriate learning rates for matrix optimization
- Implement momentum and adaptive methods for matrix parameters
- Analyze convergence of matrix optimization algorithms

**Guidance:** Students should understand how to extend gradient descent to matrix-valued parameters, with updates of the form X = X - α∇f(X). They should be able to implement gradient descent for matrix optimization problems, including choosing appropriate learning rates and initialization strategies. Extensions like momentum (incorporating previous update directions) and adaptive methods (adjusting learning rates per parameter) should be applied to matrix parameters. Students should analyze convergence properties of these algorithms and understand challenges like ill-conditioning in matrix optimization.

**6.3.8 Solve constrained optimization problems with matrix variables**
- Formulate constrained optimization problems with matrix variables
- Apply Lagrange multiplier methods to matrix constraints
- Implement projected gradient descent for matrix constraints
- Solve matrix optimization problems with orthogonality and rank constraints

**Guidance:** Students should understand how to formulate constrained optimization problems involving matrix variables, such as those with orthogonality constraints (XᵀX = I) or rank constraints. They should be able to apply Lagrange multiplier methods to derive optimality conditions for these problems. Projected gradient descent, where gradients are projected onto the feasible set after each update, should be implemented for matrix constraints. Students should practice solving specific matrix optimization problems with constraints commonly encountered in machine learning, such as orthogonal Procrustes, low-rank approximation, and semidefinite programming.

**6.3.9 Apply matrix Newton methods**
- Formulate Newton's method for matrix-valued optimization
- Compute matrix Hessian and solve Newton systems
- Implement quasi-Newton methods for matrix optimization
- Apply second-order optimization to neural network training

**Guidance:** Students should understand how to extend Newton's method to optimization problems with matrix variables, involving second derivatives. They should be able to compute matrix Hessians and solve the resulting linear systems for Newton updates. The computational challenges of exact Newton methods for large matrix variables should be recognized, leading to quasi-Newton approximations like BFGS and L-BFGS. Students should implement these methods and apply them to neural network training, comparing their convergence properties with first-order methods.

**6.3.10 Compute gradients for neural network layers**
- Derive gradients for fully connected layers
- Compute gradients for convolutional layers
- Calculate gradients for normalization layers
- Apply matrix calculus to derive gradients for custom layers

**Guidance:** Students should be able to derive gradients for standard neural network layers using matrix calculus. For fully connected layers, they should derive gradients with respect to weights and biases using the chain rule. For convolutional layers, they should understand how to compute gradients using im2col transformations or direct convolution operations. For normalization layers (batch norm, layer norm), they should derive gradients with respect to inputs and parameters. Students should practice applying matrix calculus to derive gradients for custom layer implementations, understanding how to handle element-wise operations, matrix multiplications, and other operations common in neural networks.

**6.3.11 Understand the mathematical foundation of backpropagation**
- Formulate backpropagation as an application of the chain rule
- Derive the backpropagation algorithm using matrix calculus
- Identify computational graphs and their role in automatic differentiation
- Analyze the computational complexity of backpropagation

**Guidance:** Students should understand backpropagation as an efficient application of the chain rule for computing gradients in computational graphs. They should be able to derive the backpropagation algorithm using matrix calculus, showing how gradients flow backward through the network. The concept of computational graphs representing the forward computation should be mastered, with students able to construct these graphs for neural networks. The computational complexity of backpropagation should be analyzed, showing why it's more efficient than numerical differentiation. The connection between backpropagation and automatic differentiation should be explored.

**6.3.12 Compute derivatives of common loss functions**
- Derive gradients for regression loss functions (MSE, MAE, Huber)
- Compute gradients for classification loss functions (cross-entropy, hinge)
- Calculate derivatives of regularization terms (L1, L2, elastic net)
- Apply matrix calculus to derive gradients for custom loss functions

**Guidance:** Students should be able to derive gradients for common loss functions used in machine learning. For regression losses like MSE, MAE, and Huber, they should compute gradients with respect to model predictions and parameters. For classification losses like cross-entropy and hinge loss, they should derive gradients for both binary and multi-class cases. Derivatives of regularization terms (L1, L2, elastic net) should be computed with respect to model parameters. Students should practice applying matrix calculus to derive gradients for custom loss functions, understanding how to handle different parameter structures and model architectures.

**6.3.13 Implement automatic differentiation concepts**
- Understand forward and reverse mode automatic differentiation
- Implement computational graphs for matrix operations
- Apply automatic differentiation to neural network training
- Analyze the efficiency of automatic differentiation approaches

**Guidance:** Students should understand the principles of automatic differentiation, distinguishing between forward mode (propagating derivatives from inputs to outputs) and reverse mode (propagating derivatives from outputs to inputs). They should be able to implement computational graphs for matrix operations, tracking both values and derivatives. The application of automatic differentiation to neural network training should be explored, with students implementing basic automatic differentiation systems for simple neural networks. The computational efficiency of different automatic differentiation approaches should be analyzed, particularly why reverse mode is more efficient for functions with many inputs and few outputs (like neural networks).



# Topic 4: Tensor Algebra and Operations

Students will be assessed on their ability to:

**6.4.1 Define tensors and their mathematical properties**
- Define tensors as multi-dimensional arrays with specific mathematical properties
- Distinguish between tensors and other mathematical objects (scalars, vectors, matrices)
- Understand coordinate transformations and tensor transformation rules
- Identify covariant and contravariant tensors and their properties

**Guidance:** Students should understand tensors as mathematical objects that generalize scalars, vectors, and matrices to higher dimensions. They should recognize that tensors are defined by how their components transform under coordinate changes, following specific transformation rules. The distinction between covariant (lower index) and contravariant (upper index) tensors should be clear, with students able to identify which type a tensor belongs to based on its transformation properties. Applications in physics and machine learning should be referenced.

**6.4.2 Understand tensor order, dimensions, and rank**
- Define tensor order (number of dimensions/indices) and its significance
- Understand tensor dimensions and their interpretation in different contexts
- Distinguish between tensor order and tensor rank (minimum number of components)
- Calculate the number of elements in tensors of different orders and dimensions

**Guidance:** Students should understand that tensor order refers to the number of indices required to uniquely select each element (0-order: scalar, 1-order: vector, 2-order: matrix, etc.). They should be able to interpret the meaning of different dimensions in practical contexts (e.g., height, width, channels for image tensors). The important distinction between order and rank should be emphasized, with rank being the minimum number of simple tensors needed to express the tensor. Students should practice calculating the total number of elements in tensors of various orders and dimensions.

**6.4.3 Apply tensor notation and indexing conventions**
- Use Einstein summation convention for tensor operations
- Apply multi-dimensional indexing for tensor element access
- Understand index notation for tensor expressions
- Convert between different tensor notations and representations

**Guidance:** Students should master Einstein summation convention, where repeated indices imply summation. They should be able to use multi-dimensional indexing to access specific elements in tensors, understanding the conventions for ordering indices. Index notation for expressing tensor operations should be understood, with students able to write and interpret tensor expressions using this notation. The ability to convert between different representations (index notation, Einstein notation, graphical notation) should be developed.

**6.4.4 Represent multi-dimensional data as tensors**
- Convert different data types into tensor representations
- Structure multi-modal data as higher-order tensors
- Apply appropriate tensor shapes for different data types
- Implement data preprocessing for tensor representation

**Guidance:** Students should be able to represent various types of data as tensors, including images (3D tensors: height × width × channels), videos (4D tensors: time × height × width × channels), and multi-sensor data. They should understand how to structure multi-modal data as higher-order tensors and select appropriate tensor shapes for different applications. Practical implementation of data preprocessing steps to convert raw data into tensor representations should be mastered, including normalization, reshaping, and handling missing data.

**6.4.5 Perform basic tensor manipulations and reshaping operations**
- Implement tensor reshaping and permutation operations
- Apply tensor slicing and indexing operations
- Perform tensor concatenation and stacking operations
- Execute tensor broadcasting and element-wise operations

**Guidance:** Students should be able to implement fundamental tensor operations including reshaping (changing dimensions while preserving elements), permutation (reordering dimensions), and transposition (swapping dimensions). They should master tensor slicing (extracting subsets) and indexing (accessing specific elements or regions). Operations for combining tensors such as concatenation (joining along existing dimensions) and stacking (joining along new dimensions) should be implemented. Students should understand tensor broadcasting (expanding dimensions for element-wise operations) and perform element-wise arithmetic operations on tensors.

**6.4.6 Define CANDECOMP/PARAFAC (CP) decomposition**
- Formulate CP decomposition as a sum of rank-one tensors
- Understand the mathematical properties of CP decomposition
- Implement algorithms for computing CP decomposition
- Analyze the uniqueness and identifiability of CP decompositions

**Guidance:** Students should understand CP decomposition as expressing a tensor as a sum of rank-one tensors, each being the outer product of vectors. They should recognize the mathematical formulation of CP decomposition and its properties, including the challenges of computing it (NP-hard in general). Students should be able to implement basic algorithms for computing CP decomposition, such as alternating least squares (ALS). The uniqueness properties of CP decomposition and conditions under which it is identifiable should be analyzed, with students understanding when CP decomposition provides a unique representation compared to matrix factorizations.

**6.4.7 Understand Tucker decomposition and core tensors**
- Define Tucker decomposition and its relationship to higher-order SVD
- Understand the role of core tensors in Tucker decomposition
- Implement Tucker decomposition algorithms
- Apply Tucker decomposition for dimensionality reduction and feature extraction

**Guidance:** Students should understand Tucker decomposition as a higher-order generalization of SVD, decomposing a tensor into a core tensor multiplied by factor matrices along each mode. They should recognize the core tensor as capturing the interactions between different components and the factor matrices as representing the principal components in each mode. Students should be able to implement algorithms for computing Tucker decomposition, such as higher-order orthogonal iteration (HOOI). Applications to dimensionality reduction and feature extraction should be explored, with students understanding how to select appropriate ranks for the core tensor.

**6.4.8 Apply tensor train decomposition**
- Define tensor train decomposition and its mathematical formulation
- Understand the benefits of tensor train for high-dimensional data
- Implement tensor train decomposition algorithms
- Apply tensor train decomposition to compress neural networks

**Guidance:** Students should understand tensor train (TT) decomposition as representing a high-order tensor as a chain of lower-order tensors, with each core tensor connected to its neighbors. They should recognize the benefits of TT decomposition for handling the curse of dimensionality in high-dimensional data, with its linear scaling in dimension order rather than exponential. Students should be able to implement algorithms for computing TT decomposition, such as TT-SVD. Applications to compressing neural networks by representing weight tensors in TT format should be explored, with students understanding the trade-offs between compression ratio and model accuracy.

**6.4.9 Implement tensor decompositions for dimensionality reduction**
- Apply tensor decompositions to reduce dimensionality of multi-way data
- Evaluate the quality of dimensionality reduction using different metrics
- Compare tensor-based dimensionality reduction with matrix-based methods
- Implement tensor-based feature extraction techniques

**Guidance:** Students should be able to apply various tensor decompositions (CP, Tucker, TT) to reduce the dimensionality of multi-way data while preserving important information. They should implement and evaluate different metrics for assessing the quality of dimensionality reduction, such as reconstruction error, explained variance, and preservation of local or global structure. A comparative analysis of tensor-based and matrix-based dimensionality reduction methods should be conducted, highlighting the advantages of tensor approaches for capturing multi-linear relationships. Students should implement tensor-based feature extraction techniques and apply them to real-world datasets.

**6.4.10 Use tensor decompositions for data compression and feature extraction**
- Apply tensor decompositions to compress multi-dimensional data
- Evaluate compression ratios and reconstruction quality
- Extract interpretable features from tensor decompositions
- Implement tensor-based compression for specific applications

**Guidance:** Students should be able to apply tensor decompositions to compress various types of multi-dimensional data, including images, videos, and scientific data. They should evaluate the compression ratios achieved and the quality of reconstruction using appropriate metrics. The extraction of interpretable features from tensor decompositions should be mastered, with students understanding how to interpret the components in the context of the original data. Implementation of tensor-based compression for specific applications such as hyperspectral imaging, video compression, or recommender systems should be demonstrated.

**6.4.11 Define tensor multiplication operations**
- Implement tensor-tensor multiplication (tensor product)
- Apply tensor-matrix multiplication (mode product)
- Understand element-wise tensor operations
- Perform tensor outer products and inner products

**Guidance:** Students should be able to implement various tensor multiplication operations, including tensor-tensor multiplication (generalizing matrix multiplication to higher dimensions) and tensor-matrix multiplication (mode product, which multiplies a tensor by a matrix along a specific mode). They should understand element-wise tensor operations (Hadamard product) and when to use them. The implementation of tensor outer products (generalizing the outer product of vectors) and inner products (generalizing the dot product) should be mastered, with students understanding their mathematical properties and computational complexity.

**6.4.12 Understand tensor contractions and index operations**
- Define tensor contractions and their mathematical properties
- Apply Einstein summation convention for tensor contractions
- Implement efficient algorithms for tensor contractions
- Use tensor contractions in complex mathematical expressions

**Guidance:** Students should understand tensor contractions as operations that sum over one or more indices of a tensor, reducing its order. They should master the application of Einstein summation convention for expressing tensor contractions concisely. Implementation of efficient algorithms for computing tensor contractions should be explored, considering different orderings of operations to minimize computational cost. Students should be able to use tensor contractions to express complex mathematical operations commonly found in machine learning and physics, recognizing how contractions simplify the representation of these operations.

**6.4.13 Apply tensor networks and their representations**
- Define tensor networks and their graphical representations
- Understand different types of tensor network architectures
- Implement tensor network contractions and simplifications
- Apply tensor networks to efficiently represent complex operations

**Guidance:** Students should understand tensor networks as graphical representations of complex tensor operations, where tensors are nodes and contractions are edges. They should learn about different types of tensor network architectures, including matrix product states (MPS), projected entangled pair states (PEPS), and tree tensor networks. Implementation of algorithms for contracting and simplifying tensor networks should be mastered, with students understanding how to optimize the contraction order for efficiency. Applications of tensor networks to efficiently represent complex operations in quantum mechanics, statistical physics, and machine learning should be explored.

**6.4.14 Implement efficient tensor computations**
- Optimize tensor operations for computational efficiency
- Apply parallel computing techniques to tensor operations
- Implement memory-efficient tensor algorithms
- Use specialized libraries for tensor computations

**Guidance:** Students should understand techniques for optimizing tensor operations, including reordering operations to minimize intermediate tensor sizes and exploiting tensor structure. They should be able to apply parallel computing techniques to tensor operations, distributing computations across multiple processors or GPUs. Implementation of memory-efficient tensor algorithms that avoid storing large intermediate tensors should be mastered. Students should gain proficiency with specialized libraries for tensor computations such as TensorFlow, PyTorch, NumPy, and more specialized tensor libraries like TensorLy and h5py, understanding their strengths and limitations.

**6.4.15 Use tensor operations in mathematical modeling**
- Formulate mathematical models using tensor notation
- Apply tensor operations to solve differential equations
- Implement tensor-based methods for scientific computing
- Use tensor representations for physical systems

**Guidance:** Students should be able to formulate mathematical models in various fields using tensor notation, recognizing how tensors naturally represent multi-dimensional relationships. They should apply tensor operations to solve differential equations, particularly partial differential equations where tensors can represent fields or solutions in multiple dimensions. Implementation of tensor-based methods for scientific computing, such as finite element methods or spectral methods, should be mastered. Students should understand how to use tensor representations for physical systems, such as stress tensors in continuum mechanics, electromagnetic field tensors, or diffusion tensors in medical imaging.

**6.4.16 Represent multi-modal data using tensors**
- Structure heterogeneous multi-modal data as tensors
- Handle missing data in multi-modal tensor representations
- Apply tensor fusion techniques for integrating multi-modal information
- Implement tensor-based multi-modal learning algorithms

**Guidance:** Students should be able to structure heterogeneous multi-modal data (combining text, images, audio, etc.) as tensors, determining appropriate tensor structures for different types of multi-modal datasets. They should understand techniques for handling missing data in multi-modal tensor representations, including imputation methods and tensor completion algorithms. Application of tensor fusion techniques for integrating information from different modalities should be mastered, with students understanding how to weight and combine information from different sources. Implementation of tensor-based multi-modal learning algorithms for tasks like multi-modal classification, regression, or clustering should be demonstrated.

**6.4.17 Apply tensor operations in convolutional neural networks**
- Formulate convolutions as tensor operations
- Implement efficient tensor-based convolution algorithms
- Understand tensor operations in pooling and normalization layers
- Optimize tensor computations for CNN training and inference

**Guidance:** Students should understand how to formulate convolutions as tensor operations, recognizing that convolutions can be expressed as tensor contractions. They should implement efficient tensor-based convolution algorithms, including im2col approaches and direct convolution implementations. The tensor operations underlying pooling and normalization layers in CNNs should be understood, with students able to express these operations using tensor notation. Techniques for optimizing tensor computations for CNN training and inference should be explored, including memory layout optimizations and parallelization strategies.

**6.4.18 Understand tensor methods in attention mechanisms**
- Formulate attention mechanisms using tensor operations
- Implement self-attention and multi-head attention as tensor contractions
- Understand tensor operations in transformer architectures
- Optimize tensor computations for attention-based models

**Guidance:** Students should be able to formulate attention mechanisms using tensor operations, particularly tensor contractions. They should implement self-attention and multi-head attention as tensor contractions, understanding how these operations can be expressed efficiently using tensor notation. The tensor operations underlying transformer architectures should be mastered, with students able to identify the key tensor computations in these models. Techniques for optimizing tensor computations for attention-based models should be explored, including memory-efficient attention implementations and strategies for handling long sequences.

**6.4.19 Implement tensor-based neural network architectures**
- Design neural network layers using tensor operations
- Implement tensor-based layers beyond standard convolutions
- Apply tensor decompositions to compress neural network layers
- Create novel tensor-based neural network architectures

**Guidance:** Students should be able to design neural network layers using tensor operations, moving beyond standard matrix-based operations to leverage the full power of tensor algebra. They should implement tensor-based layers that go beyond standard convolutions, such as tensor contraction layers, tensor regression layers, or tensor fusion layers. Application of tensor decompositions to compress neural network layers should be mastered, with students understanding how to replace weight tensors with decomposed versions to reduce memory footprint and computational cost. Students should create novel tensor-based neural network architectures that exploit multi-linear relationships in data.

**6.4.20 Optimize tensor computations for deep learning efficiency**
- Apply memory layout optimizations for tensor operations
- Implement tensor operation fusion for computational efficiency
- Use tensor decomposition techniques to reduce model size
- Apply quantization and pruning techniques to tensor operations

**Guidance:** Students should understand memory layout optimizations for tensor operations, including different storage formats (dense, sparse, compressed) and how they affect computational efficiency. They should implement tensor operation fusion techniques that combine multiple operations into single kernels to reduce memory bandwidth requirements. Application of tensor decomposition techniques to reduce model size should be mastered, with students understanding how to select appropriate decomposition methods for different types of layers. Quantization and pruning techniques specifically designed for tensor operations should be applied, with students understanding how to maintain model accuracy while reducing computational requirements.

# Topic 5: Advanced Vector Spaces and Linear Transformations

Students will be assessed on their ability to:

**6.5.1 Define vector spaces axiomatically**
- State and verify the ten vector space axioms
- Prove properties derived from vector space axioms
- Identify examples and non-examples of vector spaces
- Apply axiomatic reasoning to vector space problems

**Guidance:** Students should master the ten axioms defining a vector space: closure under addition, commutativity of addition, associativity of addition, existence of additive identity, existence of additive inverses, closure under scalar multiplication, compatibility of scalar multiplication with field multiplication, identity element of scalar multiplication, and distributivity of scalar multiplication with respect to vector addition and scalar addition. They should be able to prove basic properties derived from these axioms, such as the uniqueness of the zero vector and the fact that -v = (-1)v. Students should identify various examples of vector spaces (R^n, function spaces, matrix spaces) and recognize sets that fail to satisfy vector space axioms. Applications of axiomatic reasoning to solve problems in linear algebra should be emphasized.

**6.5.2 Analyze subspaces and their properties**
- Define subspaces and identify subspace criteria
- Prove subsets are subspaces using subspace tests
- Find intersections, sums, and direct sums of subspaces
- Apply subspace concepts to solution spaces of linear systems

**Guidance:** Students should understand a subspace as a subset of a vector space that is itself a vector space under the same operations. They should master the subspace test: a non-empty subset W is a subspace if and only if it is closed under vector addition and scalar multiplication. Students should be able to prove subsets are subspaces using this test and find intersections, sums, and direct sums of subspaces. The concepts of linear combinations and span should be connected to subspaces. Applications to solution spaces of homogeneous linear systems should be explored, with students recognizing that these solutions form subspaces.

**6.5.3 Determine basis and dimension of vector spaces**
- Define linear independence and dependence
- Determine if sets of vectors form a basis
- Find bases for vector spaces and subspaces
- Calculate dimension and understand its invariance

**Guidance:** Students should understand linear independence as a set of vectors where no vector can be expressed as a linear combination of the others. They should be able to determine if sets of vectors form a basis (linearly independent spanning sets) and find bases for various vector spaces and subspaces. The concept of dimension as the number of vectors in a basis should be mastered, with students understanding that dimension is invariant (all bases have the same number of elements). Techniques for extending linearly independent sets to bases and reducing spanning sets to bases should be implemented. Applications to finding bases for solution spaces and column spaces should be explored.

**6.5.4 Apply abstract vector space theory to function spaces**
- Identify function spaces as vector spaces
- Determine bases for polynomial spaces
- Find dimensions of function spaces
- Apply vector space concepts to solve problems in function spaces

**Guidance:** Students should recognize various function spaces as vector spaces, including spaces of polynomials, continuous functions, and differentiable functions. They should be able to determine standard bases for polynomial spaces (monomial basis, Legendre polynomials, etc.) and find dimensions of finite-dimensional function spaces. The concepts of linear independence, span, and basis should be applied to function spaces, with students able to determine if a set of functions is linearly independent using techniques like the Wronskian. Applications to solving differential equations and approximation theory should be explored.

**6.5.5 Define linear transformations between vector spaces**
- State the definition and properties of linear transformations
- Verify if functions between vector spaces are linear
- Find the matrix representation of linear transformations
- Apply linear transformation concepts to feature mapping

**Guidance:** Students should understand linear transformations as functions T: V → W between vector spaces that preserve vector addition and scalar multiplication (T(u+v) = T(u) + T(v) and T(cv) = cT(v)). They should be able to verify if given functions are linear transformations and prove basic properties of linear transformations. The matrix representation of linear transformations with respect to given bases should be mastered, with students able to find the matrix representation and compute the transformation using matrix multiplication. Applications to feature mapping in machine learning, particularly in kernel methods, should be explored.

**6.5.6 Analyze null space and range of linear transformations**
- Define null space (kernel) and range (image) of linear transformations
- Prove null space and range are subspaces
- Find bases for null space and range
- Apply the rank-nullity theorem to analyze linear transformations

**Guidance:** Students should understand the null space (kernel) as the set of vectors that map to zero and the range (image) as the set of all outputs of a linear transformation. They should be able to prove that both are subspaces and find bases for these subspaces. The rank-nullity theorem (dim(null(T)) + dim(range(T)) = dim(V)) should be mastered, with students applying it to analyze properties of linear transformations. The relationship between null space and range concepts and solutions to linear systems should be explored. Applications to understanding the behavior of neural network layers and feature transformations should be emphasized.

**6.5.7 Define inner products and their properties**
- State the definition and properties of inner products
- Verify if functions are inner products
- Understand the Cauchy-Schwarz inequality and its applications
- Apply inner product concepts to machine learning similarity measures

**Guidance:** Students should understand inner products as functions that take two vectors and return a scalar, satisfying conjugate symmetry, linearity in the first argument, and positive-definiteness. They should be able to verify if given functions are inner products and prove basic properties derived from the inner product axioms. The Cauchy-Schwarz inequality (|⟨u,v⟩|² ≤ ⟨u,u⟩⟨v,v⟩) and its applications should be mastered. Various examples of inner products (standard dot product, weighted inner products, function space inner products) should be analyzed. Applications to similarity measures and distance metrics in machine learning should be explored.

**6.5.8 Understand norms induced by inner products**
- Define norms induced by inner products
- Verify the parallelogram law for inner product spaces
- Understand the relationship between norms and inner products
- Apply norm concepts to measure vector magnitudes in ML applications

**Guid guidance:** Students should understand how inner products induce norms through the formula ||v|| = √⟨v,v⟩. They should be able to verify the parallelogram law (||u+v||² + ||u-v||² = 2(||u||² + ||v||²)) and recognize it as a characterization of norms induced by inner products. The relationship between norms and inner products, including the polarization formula, should be mastered. Various examples of norms induced by different inner products should be analyzed. Applications to measuring vector magnitudes and distances in machine learning applications should be explored, particularly in clustering and classification algorithms.

**6.5.9 Analyze orthogonality and orthonormal bases**
- Define orthogonal and orthonormal sets of vectors
- Apply the Gram-Schmidt process to construct orthonormal bases
- Find orthogonal projections onto subspaces
- Apply orthogonality concepts to feature extraction and dimensionality reduction

**Guidance:** Students should understand orthogonal sets as sets of vectors where each pair is orthogonal (⟨u,v⟩ = 0) and orthonormal sets as orthogonal sets where each vector has unit length. They should be able to apply the Gram-Schmidt process to construct orthonormal bases from linearly independent sets. The concept of orthogonal projection onto subspaces should be mastered, with students able to compute projections and decompose vectors into orthogonal components. Applications to feature extraction and dimensionality reduction techniques like PCA should be explored, with students understanding how orthogonality helps in creating uncorrelated features.

**6.5.10 Define adjoint operators and their properties**
- Define adjoint operators in inner product spaces
- Find matrix representations of adjoint operators
- Prove properties of adjoint operators
- Apply adjoint operators to solve optimization problems

**Guidance:** Students should understand the adjoint T* of a linear operator T as the unique operator satisfying ⟨T(u),v⟩ = ⟨u,T*(v)⟩ for all vectors u,v. They should be able to find matrix representations of adjoint operators (conjugate transpose for complex matrices, simple transpose for real matrices). Basic properties of adjoint operators, such as (T*)* = T, (cT)* = c̅T*, and (T+S)* = T* + S*, should be proven. Applications to solving optimization problems, particularly least squares problems, should be explored. The connection between adjoint operators and gradient computations in machine learning should be emphasized.

**6.5.11 Understand projection operators and orthogonal projections**
- Define projection operators and their properties
- Distinguish between orthogonal and oblique projections
- Find matrix representations of projection operators
- Apply projection operators to solve approximation problems

**Guidance:** Students should understand projection operators as linear transformations P that satisfy P² = P (idempotent). They should distinguish between orthogonal projections (where range(P) ⊥ null(P)) and oblique projections (where range(P) and null(P) are not necessarily orthogonal). The matrix representations of projection operators should be found, particularly for orthogonal projections onto subspaces. Applications to solving approximation problems, such as finding the best approximation of a vector from a subspace, should be explored. The use of projection operators in iterative methods like Krylov subspace methods should be referenced.

**6.5.12 Analyze spectral properties of linear operators**
- Define eigenvalues and eigenvectors of linear operators
- Understand the spectrum of linear operators
- Apply the spectral theorem to self-adjoint operators
- Analyze spectral properties of special classes of operators

**Guidance:** Students should understand eigenvalues and eigenvectors of linear operators as generalizations of matrix eigenvalues and eigenvectors, defined by the equation T(v) = λv. The spectrum of a linear operator should be defined as the set of all eigenvalues. The spectral theorem for self-adjoint operators, which states that self-adjoint operators have an orthonormal basis of eigenvectors and real eigenvalues, should be mastered. Students should analyze spectral properties of special classes of operators, including normal operators, unitary operators, and positive definite operators. Applications to understanding the behavior of linear transformations in machine learning models should be explored.

**6.5.13 Apply advanced transformation concepts to dimensionality reduction**
- Understand principal component analysis through spectral theory
- Apply linear transformation concepts to nonlinear dimensionality reduction
- Use spectral properties to analyze manifold learning techniques
- Implement advanced dimensionality reduction methods using linear algebra

**Guidance:** Students should understand principal component analysis (PCA) through the lens of spectral theory, recognizing it as finding an orthogonal basis that maximizes variance. They should apply linear transformation concepts to nonlinear dimensionality reduction techniques like kernel PCA, which maps data to higher-dimensional spaces where linear methods can be applied. The use of spectral properties in analyzing manifold learning techniques like Laplacian Eigenmaps should be explored. Implementation of advanced dimensionality reduction methods using linear algebra concepts should be mastered, with students understanding the underlying mathematical foundations.

**6.5.14 Apply advanced vector space concepts to functional analysis**
- Define Banach spaces and Hilbert spaces
- Understand convergence in infinite-dimensional spaces
- Apply concepts of orthogonality to function spaces
- Use functional analysis concepts in machine learning theory

**Guidance:** Students should understand Banach spaces as complete normed vector spaces and Hilbert spaces as complete inner product spaces. They should understand concepts of convergence in infinite-dimensional spaces, including norm convergence and weak convergence. The application of orthogonality concepts to function spaces, particularly in the context of Fourier series and orthogonal polynomials, should be mastered. The use of functional analysis concepts in machine learning theory, particularly in understanding the properties of kernel methods and regularization, should be explored. Students should recognize how functional analysis provides a rigorous foundation for many machine learning techniques.

