??:
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.6.02unknown8??8
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
??*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	?*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_29/lstm_cell_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_29/lstm_cell_29/kernel
?
/lstm_29/lstm_cell_29/kernel/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_29/lstm_cell_29/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_29/lstm_cell_29/recurrent_kernel
?
9lstm_29/lstm_cell_29/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_29/lstm_cell_29/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_29/lstm_cell_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_29/lstm_cell_29/bias
?
-lstm_29/lstm_cell_29/bias/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/bias*
_output_shapes	
:?*
dtype0
?
lstm_30/lstm_cell_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelstm_30/lstm_cell_30/kernel
?
/lstm_30/lstm_cell_30/kernel/Read/ReadVariableOpReadVariableOplstm_30/lstm_cell_30/kernel* 
_output_shapes
:
??*
dtype0
?
%lstm_30/lstm_cell_30/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_30/lstm_cell_30/recurrent_kernel
?
9lstm_30/lstm_cell_30/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_30/lstm_cell_30/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_30/lstm_cell_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_30/lstm_cell_30/bias
?
-lstm_30/lstm_cell_30/bias/Read/ReadVariableOpReadVariableOplstm_30/lstm_cell_30/bias*
_output_shapes	
:?*
dtype0
?
lstm_31/lstm_cell_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelstm_31/lstm_cell_31/kernel
?
/lstm_31/lstm_cell_31/kernel/Read/ReadVariableOpReadVariableOplstm_31/lstm_cell_31/kernel* 
_output_shapes
:
??*
dtype0
?
%lstm_31/lstm_cell_31/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_31/lstm_cell_31/recurrent_kernel
?
9lstm_31/lstm_cell_31/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_31/lstm_cell_31/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_31/lstm_cell_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_31/lstm_cell_31/bias
?
-lstm_31/lstm_cell_31/bias/Read/ReadVariableOpReadVariableOplstm_31/lstm_cell_31/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_32/kernel/m
?
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_29/lstm_cell_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/m
?
6Adam/lstm_29/lstm_cell_29/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
?
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_29/lstm_cell_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/m
?
4Adam/lstm_29/lstm_cell_29/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_30/lstm_cell_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_30/lstm_cell_30/kernel/m
?
6Adam/lstm_30/lstm_cell_30/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_30/lstm_cell_30/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_30/lstm_cell_30/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m
?
@Adam/lstm_30/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_30/lstm_cell_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_30/lstm_cell_30/bias/m
?
4Adam/lstm_30/lstm_cell_30/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_30/lstm_cell_30/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_31/lstm_cell_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_31/lstm_cell_31/kernel/m
?
6Adam/lstm_31/lstm_cell_31/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_31/lstm_cell_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_31/lstm_cell_31/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m
?
@Adam/lstm_31/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_31/lstm_cell_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_31/lstm_cell_31/bias/m
?
4Adam/lstm_31/lstm_cell_31/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_31/lstm_cell_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_32/kernel/v
?
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_29/lstm_cell_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/v
?
6Adam/lstm_29/lstm_cell_29/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
?
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_29/lstm_cell_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/v
?
4Adam/lstm_29/lstm_cell_29/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_30/lstm_cell_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_30/lstm_cell_30/kernel/v
?
6Adam/lstm_30/lstm_cell_30/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_30/lstm_cell_30/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_30/lstm_cell_30/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v
?
@Adam/lstm_30/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_30/lstm_cell_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_30/lstm_cell_30/bias/v
?
4Adam/lstm_30/lstm_cell_30/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_30/lstm_cell_30/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_31/lstm_cell_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_31/lstm_cell_31/kernel/v
?
6Adam/lstm_31/lstm_cell_31/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_31/lstm_cell_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_31/lstm_cell_31/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v
?
@Adam/lstm_31/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_31/lstm_cell_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_31/lstm_cell_31/bias/v
?
4Adam/lstm_31/lstm_cell_31/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_31/lstm_cell_31/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?U
value?UB?U B?U
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
l
$cell
%
state_spec
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.m?/m?8m?9m?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?.v?/v?8v?9v?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?
^
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912
^
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912
 
?

Llayers
trainable_variables
Mmetrics
	variables
regularization_losses
Nlayer_metrics
Olayer_regularization_losses
Pnon_trainable_variables
 
?
Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
 

C0
D1
E2

C0
D1
E2
 
?

Vlayers
trainable_variables
	variables
Wmetrics
regularization_losses
Xlayer_metrics
Ylayer_regularization_losses

Zstates
[non_trainable_variables
 
 
 
?

\layers
trainable_variables
]metrics
	variables
regularization_losses
^layer_metrics
_layer_regularization_losses
`non_trainable_variables
?
a
state_size

Fkernel
Grecurrent_kernel
Hbias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
 

F0
G1
H2

F0
G1
H2
 
?

flayers
trainable_variables
	variables
gmetrics
regularization_losses
hlayer_metrics
ilayer_regularization_losses

jstates
knon_trainable_variables
 
 
 
?

llayers
 trainable_variables
mmetrics
!	variables
"regularization_losses
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
?
q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
 

I0
J1
K2

I0
J1
K2
 
?

vlayers
&trainable_variables
'	variables
wmetrics
(regularization_losses
xlayer_metrics
ylayer_regularization_losses

zstates
{non_trainable_variables
 
 
 
?

|layers
*trainable_variables
}metrics
+	variables
,regularization_losses
~layer_metrics
layer_regularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
?layers
0trainable_variables
?metrics
1	variables
2regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
4trainable_variables
?metrics
5	variables
6regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
?layers
:trainable_variables
?metrics
;	variables
<regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_29/lstm_cell_29/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_29/lstm_cell_29/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_29/lstm_cell_29/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_30/lstm_cell_30/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_30/lstm_cell_30/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_30/lstm_cell_30/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_31/lstm_cell_31/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_31/lstm_cell_31/recurrent_kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_31/lstm_cell_31/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

?0
?1
 
 
 
 

C0
D1
E2

C0
D1
E2
 
?
?layers
Rtrainable_variables
?metrics
S	variables
Tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables

0
 
 
 
 
 
 
 
 
 
 
 

F0
G1
H2

F0
G1
H2
 
?
?layers
btrainable_variables
?metrics
c	variables
dregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables

0
 
 
 
 
 
 
 
 
 
 
 

I0
J1
K2

I0
J1
K2
 
?
?layers
rtrainable_variables
?metrics
s	variables
tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables

$0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_30/lstm_cell_30/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_30/lstm_cell_30/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_30/lstm_cell_30/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_31/lstm_cell_31/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_31/lstm_cell_31/recurrent_kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_31/lstm_cell_31/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_30/lstm_cell_30/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_30/lstm_cell_30/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_30/lstm_cell_30/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_31/lstm_cell_31/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_31/lstm_cell_31/recurrent_kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_31/lstm_cell_31/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_29_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_29_inputlstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biaslstm_30/lstm_cell_30/kernel%lstm_30/lstm_cell_30/recurrent_kernellstm_30/lstm_cell_30/biaslstm_31/lstm_cell_31/kernel%lstm_31/lstm_cell_31/recurrent_kernellstm_31/lstm_cell_31/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1258009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_29/lstm_cell_29/kernel/Read/ReadVariableOp9lstm_29/lstm_cell_29/recurrent_kernel/Read/ReadVariableOp-lstm_29/lstm_cell_29/bias/Read/ReadVariableOp/lstm_30/lstm_cell_30/kernel/Read/ReadVariableOp9lstm_30/lstm_cell_30/recurrent_kernel/Read/ReadVariableOp-lstm_30/lstm_cell_30/bias/Read/ReadVariableOp/lstm_31/lstm_cell_31/kernel/Read/ReadVariableOp9lstm_31/lstm_cell_31/recurrent_kernel/Read/ReadVariableOp-lstm_31/lstm_cell_31/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp6Adam/lstm_29/lstm_cell_29/kernel/m/Read/ReadVariableOp@Adam/lstm_29/lstm_cell_29/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_29/lstm_cell_29/bias/m/Read/ReadVariableOp6Adam/lstm_30/lstm_cell_30/kernel/m/Read/ReadVariableOp@Adam/lstm_30/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_30/lstm_cell_30/bias/m/Read/ReadVariableOp6Adam/lstm_31/lstm_cell_31/kernel/m/Read/ReadVariableOp@Adam/lstm_31/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_31/lstm_cell_31/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp6Adam/lstm_29/lstm_cell_29/kernel/v/Read/ReadVariableOp@Adam/lstm_29/lstm_cell_29/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_29/lstm_cell_29/bias/v/Read/ReadVariableOp6Adam/lstm_30/lstm_cell_30/kernel/v/Read/ReadVariableOp@Adam/lstm_30/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_30/lstm_cell_30/bias/v/Read/ReadVariableOp6Adam/lstm_31/lstm_cell_31/kernel/v/Read/ReadVariableOp@Adam/lstm_31/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_31/lstm_cell_31/bias/v/Read/ReadVariableOpConst*=
Tin6
422	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1261551
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biaslstm_30/lstm_cell_30/kernel%lstm_30/lstm_cell_30/recurrent_kernellstm_30/lstm_cell_30/biaslstm_31/lstm_cell_31/kernel%lstm_31/lstm_cell_31/recurrent_kernellstm_31/lstm_cell_31/biastotalcounttotal_1count_1Adam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/m"Adam/lstm_29/lstm_cell_29/kernel/m,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m Adam/lstm_29/lstm_cell_29/bias/m"Adam/lstm_30/lstm_cell_30/kernel/m,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m Adam/lstm_30/lstm_cell_30/bias/m"Adam/lstm_31/lstm_cell_31/kernel/m,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m Adam/lstm_31/lstm_cell_31/bias/mAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/v"Adam/lstm_29/lstm_cell_29/kernel/v,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v Adam/lstm_29/lstm_cell_29/bias/v"Adam/lstm_30/lstm_cell_30/kernel/v,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v Adam/lstm_30/lstm_cell_30/bias/v"Adam/lstm_31/lstm_cell_31/kernel/v,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v Adam/lstm_31/lstm_cell_31/bias/v*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1261705??6
?
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261039

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260341

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?^
?
(sequential_13_lstm_30_while_body_1254492H
Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counterN
Jsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations+
'sequential_13_lstm_30_while_placeholder-
)sequential_13_lstm_30_while_placeholder_1-
)sequential_13_lstm_30_while_placeholder_2-
)sequential_13_lstm_30_while_placeholder_3G
Csequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1_0?
sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
??_
Ksequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??Y
Jsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	?(
$sequential_13_lstm_30_while_identity*
&sequential_13_lstm_30_while_identity_1*
&sequential_13_lstm_30_while_identity_2*
&sequential_13_lstm_30_while_identity_3*
&sequential_13_lstm_30_while_identity_4*
&sequential_13_lstm_30_while_identity_5E
Asequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1?
}sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
??]
Isequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
??W
Hsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	????sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
Msequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2O
Msequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_30_while_placeholderVsequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02A
?sequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem?
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?
/sequential_13/lstm_30/while/lstm_cell_30/MatMulMatMulFsequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_30/while/lstm_cell_30/MatMul?
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02B
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
1sequential_13/lstm_30/while/lstm_cell_30/MatMul_1MatMul)sequential_13_lstm_30_while_placeholder_2Hsequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_13/lstm_30/while/lstm_cell_30/MatMul_1?
,sequential_13/lstm_30/while/lstm_cell_30/addAddV29sequential_13/lstm_30/while/lstm_cell_30/MatMul:product:0;sequential_13/lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_30/while/lstm_cell_30/add?
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02A
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?
0sequential_13/lstm_30/while/lstm_cell_30/BiasAddBiasAdd0sequential_13/lstm_30/while/lstm_cell_30/add:z:0Gsequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_30/while/lstm_cell_30/BiasAdd?
8sequential_13/lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_30/while/lstm_cell_30/split/split_dim?
.sequential_13/lstm_30/while/lstm_cell_30/splitSplitAsequential_13/lstm_30/while/lstm_cell_30/split/split_dim:output:09sequential_13/lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split20
.sequential_13/lstm_30/while/lstm_cell_30/split?
0sequential_13/lstm_30/while/lstm_cell_30/SigmoidSigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_30/while/lstm_cell_30/Sigmoid?
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1?
,sequential_13/lstm_30/while/lstm_cell_30/mulMul6sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1:y:0)sequential_13_lstm_30_while_placeholder_3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_30/while/lstm_cell_30/mul?
-sequential_13/lstm_30/while/lstm_cell_30/ReluRelu7sequential_13/lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2/
-sequential_13/lstm_30/while/lstm_cell_30/Relu?
.sequential_13/lstm_30/while/lstm_cell_30/mul_1Mul4sequential_13/lstm_30/while/lstm_cell_30/Sigmoid:y:0;sequential_13/lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_30/while/lstm_cell_30/mul_1?
.sequential_13/lstm_30/while/lstm_cell_30/add_1AddV20sequential_13/lstm_30/while/lstm_cell_30/mul:z:02sequential_13/lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_30/while/lstm_cell_30/add_1?
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2?
/sequential_13/lstm_30/while/lstm_cell_30/Relu_1Relu2sequential_13/lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_30/while/lstm_cell_30/Relu_1?
.sequential_13/lstm_30/while/lstm_cell_30/mul_2Mul6sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2:y:0=sequential_13/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_30/while/lstm_cell_30/mul_2?
@sequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_30_while_placeholder_1'sequential_13_lstm_30_while_placeholder2sequential_13/lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItem?
!sequential_13/lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_30/while/add/y?
sequential_13/lstm_30/while/addAddV2'sequential_13_lstm_30_while_placeholder*sequential_13/lstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_30/while/add?
#sequential_13/lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_30/while/add_1/y?
!sequential_13/lstm_30/while/add_1AddV2Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counter,sequential_13/lstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_30/while/add_1?
$sequential_13/lstm_30/while/IdentityIdentity%sequential_13/lstm_30/while/add_1:z:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_30/while/Identity?
&sequential_13/lstm_30/while/Identity_1IdentityJsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_1?
&sequential_13/lstm_30/while/Identity_2Identity#sequential_13/lstm_30/while/add:z:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_2?
&sequential_13/lstm_30/while/Identity_3IdentityPsequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_3?
&sequential_13/lstm_30/while/Identity_4Identity2sequential_13/lstm_30/while/lstm_cell_30/mul_2:z:0!^sequential_13/lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_30/while/Identity_4?
&sequential_13/lstm_30/while/Identity_5Identity2sequential_13/lstm_30/while/lstm_cell_30/add_1:z:0!^sequential_13/lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_30/while/Identity_5?
 sequential_13/lstm_30/while/NoOpNoOp@^sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?^sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpA^sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_30/while/NoOp"U
$sequential_13_lstm_30_while_identity-sequential_13/lstm_30/while/Identity:output:0"Y
&sequential_13_lstm_30_while_identity_1/sequential_13/lstm_30/while/Identity_1:output:0"Y
&sequential_13_lstm_30_while_identity_2/sequential_13/lstm_30/while/Identity_2:output:0"Y
&sequential_13_lstm_30_while_identity_3/sequential_13/lstm_30/while/Identity_3:output:0"Y
&sequential_13_lstm_30_while_identity_4/sequential_13/lstm_30/while/Identity_4:output:0"Y
&sequential_13_lstm_30_while_identity_5/sequential_13/lstm_30/while/Identity_5:output:0"?
Hsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resourceJsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"?
Isequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resourceKsequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"?
Gsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resourceIsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"?
Asequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1Csequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1_0"?
}sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2?
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2?
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_1256727

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
E__inference_dense_32_layer_call_and_return_conditional_losses_1257117

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1259586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259586___redundant_placeholder05
1while_while_cond_1259586___redundant_placeholder15
1while_while_cond_1259586___redundant_placeholder25
1while_while_cond_1259586___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_30_layer_call_fn_1259709
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12555192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
.__inference_lstm_cell_29_layer_call_fn_1261124

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12549842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?U
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260957

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260873*
condR
while_cond_1260872*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_1254771
lstm_29_inputT
Asequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resource:	?W
Csequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
??Q
Bsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	?U
Asequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resource:
??W
Csequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
??Q
Bsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	?U
Asequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resource:
??W
Csequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
??Q
Bsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	?L
8sequential_13_dense_31_tensordot_readvariableop_resource:
??E
6sequential_13_dense_31_biasadd_readvariableop_resource:	?K
8sequential_13_dense_32_tensordot_readvariableop_resource:	?D
6sequential_13_dense_32_biasadd_readvariableop_resource:
identity??-sequential_13/dense_31/BiasAdd/ReadVariableOp?/sequential_13/dense_31/Tensordot/ReadVariableOp?-sequential_13/dense_32/BiasAdd/ReadVariableOp?/sequential_13/dense_32/Tensordot/ReadVariableOp?9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp?:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?sequential_13/lstm_29/while?9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp?:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?sequential_13/lstm_30/while?9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp?:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?sequential_13/lstm_31/whilew
sequential_13/lstm_29/ShapeShapelstm_29_input*
T0*
_output_shapes
:2
sequential_13/lstm_29/Shape?
)sequential_13/lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_29/strided_slice/stack?
+sequential_13/lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_29/strided_slice/stack_1?
+sequential_13/lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_29/strided_slice/stack_2?
#sequential_13/lstm_29/strided_sliceStridedSlice$sequential_13/lstm_29/Shape:output:02sequential_13/lstm_29/strided_slice/stack:output:04sequential_13/lstm_29/strided_slice/stack_1:output:04sequential_13/lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_29/strided_slice?
$sequential_13/lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_13/lstm_29/zeros/packed/1?
"sequential_13/lstm_29/zeros/packedPack,sequential_13/lstm_29/strided_slice:output:0-sequential_13/lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_29/zeros/packed?
!sequential_13/lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_29/zeros/Const?
sequential_13/lstm_29/zerosFill+sequential_13/lstm_29/zeros/packed:output:0*sequential_13/lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_29/zeros?
&sequential_13/lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_13/lstm_29/zeros_1/packed/1?
$sequential_13/lstm_29/zeros_1/packedPack,sequential_13/lstm_29/strided_slice:output:0/sequential_13/lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_29/zeros_1/packed?
#sequential_13/lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_29/zeros_1/Const?
sequential_13/lstm_29/zeros_1Fill-sequential_13/lstm_29/zeros_1/packed:output:0,sequential_13/lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_29/zeros_1?
$sequential_13/lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_29/transpose/perm?
sequential_13/lstm_29/transpose	Transposelstm_29_input-sequential_13/lstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2!
sequential_13/lstm_29/transpose?
sequential_13/lstm_29/Shape_1Shape#sequential_13/lstm_29/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_29/Shape_1?
+sequential_13/lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_29/strided_slice_1/stack?
-sequential_13/lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_1/stack_1?
-sequential_13/lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_1/stack_2?
%sequential_13/lstm_29/strided_slice_1StridedSlice&sequential_13/lstm_29/Shape_1:output:04sequential_13/lstm_29/strided_slice_1/stack:output:06sequential_13/lstm_29/strided_slice_1/stack_1:output:06sequential_13/lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_1?
1sequential_13/lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_13/lstm_29/TensorArrayV2/element_shape?
#sequential_13/lstm_29/TensorArrayV2TensorListReserve:sequential_13/lstm_29/TensorArrayV2/element_shape:output:0.sequential_13/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_29/TensorArrayV2?
Ksequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_29/transpose:y:0Tsequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor?
+sequential_13/lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_29/strided_slice_2/stack?
-sequential_13/lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_2/stack_1?
-sequential_13/lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_2/stack_2?
%sequential_13/lstm_29/strided_slice_2StridedSlice#sequential_13/lstm_29/transpose:y:04sequential_13/lstm_29/strided_slice_2/stack:output:06sequential_13/lstm_29/strided_slice_2/stack_1:output:06sequential_13/lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_2?
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp?
)sequential_13/lstm_29/lstm_cell_29/MatMulMatMul.sequential_13/lstm_29/strided_slice_2:output:0@sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_29/lstm_cell_29/MatMul?
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?
+sequential_13/lstm_29/lstm_cell_29/MatMul_1MatMul$sequential_13/lstm_29/zeros:output:0Bsequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_13/lstm_29/lstm_cell_29/MatMul_1?
&sequential_13/lstm_29/lstm_cell_29/addAddV23sequential_13/lstm_29/lstm_cell_29/MatMul:product:05sequential_13/lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_29/lstm_cell_29/add?
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?
*sequential_13/lstm_29/lstm_cell_29/BiasAddBiasAdd*sequential_13/lstm_29/lstm_cell_29/add:z:0Asequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_29/lstm_cell_29/BiasAdd?
2sequential_13/lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_29/lstm_cell_29/split/split_dim?
(sequential_13/lstm_29/lstm_cell_29/splitSplit;sequential_13/lstm_29/lstm_cell_29/split/split_dim:output:03sequential_13/lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2*
(sequential_13/lstm_29/lstm_cell_29/split?
*sequential_13/lstm_29/lstm_cell_29/SigmoidSigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_29/lstm_cell_29/Sigmoid?
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_1Sigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_1?
&sequential_13/lstm_29/lstm_cell_29/mulMul0sequential_13/lstm_29/lstm_cell_29/Sigmoid_1:y:0&sequential_13/lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_29/lstm_cell_29/mul?
'sequential_13/lstm_29/lstm_cell_29/ReluRelu1sequential_13/lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2)
'sequential_13/lstm_29/lstm_cell_29/Relu?
(sequential_13/lstm_29/lstm_cell_29/mul_1Mul.sequential_13/lstm_29/lstm_cell_29/Sigmoid:y:05sequential_13/lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_29/lstm_cell_29/mul_1?
(sequential_13/lstm_29/lstm_cell_29/add_1AddV2*sequential_13/lstm_29/lstm_cell_29/mul:z:0,sequential_13/lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_29/lstm_cell_29/add_1?
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_2Sigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_2?
)sequential_13/lstm_29/lstm_cell_29/Relu_1Relu,sequential_13/lstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_29/lstm_cell_29/Relu_1?
(sequential_13/lstm_29/lstm_cell_29/mul_2Mul0sequential_13/lstm_29/lstm_cell_29/Sigmoid_2:y:07sequential_13/lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_29/lstm_cell_29/mul_2?
3sequential_13/lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   25
3sequential_13/lstm_29/TensorArrayV2_1/element_shape?
%sequential_13/lstm_29/TensorArrayV2_1TensorListReserve<sequential_13/lstm_29/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_29/TensorArrayV2_1z
sequential_13/lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_29/time?
.sequential_13/lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_13/lstm_29/while/maximum_iterations?
(sequential_13/lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_29/while/loop_counter?
sequential_13/lstm_29/whileWhile1sequential_13/lstm_29/while/loop_counter:output:07sequential_13/lstm_29/while/maximum_iterations:output:0#sequential_13/lstm_29/time:output:0.sequential_13/lstm_29/TensorArrayV2_1:handle:0$sequential_13/lstm_29/zeros:output:0&sequential_13/lstm_29/zeros_1:output:0.sequential_13/lstm_29/strided_slice_1:output:0Msequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resourceCsequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resourceBsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_29_while_body_1254352*4
cond,R*
(sequential_13_lstm_29_while_cond_1254351*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_13/lstm_29/while?
Fsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2H
Fsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential_13/lstm_29/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_29/while:output:3Osequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02:
8sequential_13/lstm_29/TensorArrayV2Stack/TensorListStack?
+sequential_13/lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential_13/lstm_29/strided_slice_3/stack?
-sequential_13/lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_29/strided_slice_3/stack_1?
-sequential_13/lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_3/stack_2?
%sequential_13/lstm_29/strided_slice_3StridedSliceAsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_29/strided_slice_3/stack:output:06sequential_13/lstm_29/strided_slice_3/stack_1:output:06sequential_13/lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_3?
&sequential_13/lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_29/transpose_1/perm?
!sequential_13/lstm_29/transpose_1	TransposeAsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/lstm_29/transpose_1?
sequential_13/lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_29/runtime?
!sequential_13/dropout_47/IdentityIdentity%sequential_13/lstm_29/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/dropout_47/Identity?
sequential_13/lstm_30/ShapeShape*sequential_13/dropout_47/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_30/Shape?
)sequential_13/lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_30/strided_slice/stack?
+sequential_13/lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_30/strided_slice/stack_1?
+sequential_13/lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_30/strided_slice/stack_2?
#sequential_13/lstm_30/strided_sliceStridedSlice$sequential_13/lstm_30/Shape:output:02sequential_13/lstm_30/strided_slice/stack:output:04sequential_13/lstm_30/strided_slice/stack_1:output:04sequential_13/lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_30/strided_slice?
$sequential_13/lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_13/lstm_30/zeros/packed/1?
"sequential_13/lstm_30/zeros/packedPack,sequential_13/lstm_30/strided_slice:output:0-sequential_13/lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_30/zeros/packed?
!sequential_13/lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_30/zeros/Const?
sequential_13/lstm_30/zerosFill+sequential_13/lstm_30/zeros/packed:output:0*sequential_13/lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_30/zeros?
&sequential_13/lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_13/lstm_30/zeros_1/packed/1?
$sequential_13/lstm_30/zeros_1/packedPack,sequential_13/lstm_30/strided_slice:output:0/sequential_13/lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_30/zeros_1/packed?
#sequential_13/lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_30/zeros_1/Const?
sequential_13/lstm_30/zeros_1Fill-sequential_13/lstm_30/zeros_1/packed:output:0,sequential_13/lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_30/zeros_1?
$sequential_13/lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_30/transpose/perm?
sequential_13/lstm_30/transpose	Transpose*sequential_13/dropout_47/Identity:output:0-sequential_13/lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2!
sequential_13/lstm_30/transpose?
sequential_13/lstm_30/Shape_1Shape#sequential_13/lstm_30/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_30/Shape_1?
+sequential_13/lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_30/strided_slice_1/stack?
-sequential_13/lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_1/stack_1?
-sequential_13/lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_1/stack_2?
%sequential_13/lstm_30/strided_slice_1StridedSlice&sequential_13/lstm_30/Shape_1:output:04sequential_13/lstm_30/strided_slice_1/stack:output:06sequential_13/lstm_30/strided_slice_1/stack_1:output:06sequential_13/lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_1?
1sequential_13/lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_13/lstm_30/TensorArrayV2/element_shape?
#sequential_13/lstm_30/TensorArrayV2TensorListReserve:sequential_13/lstm_30/TensorArrayV2/element_shape:output:0.sequential_13/lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_30/TensorArrayV2?
Ksequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_30/transpose:y:0Tsequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor?
+sequential_13/lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_30/strided_slice_2/stack?
-sequential_13/lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_2/stack_1?
-sequential_13/lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_2/stack_2?
%sequential_13/lstm_30/strided_slice_2StridedSlice#sequential_13/lstm_30/transpose:y:04sequential_13/lstm_30/strided_slice_2/stack:output:06sequential_13/lstm_30/strided_slice_2/stack_1:output:06sequential_13/lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_2?
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp?
)sequential_13/lstm_30/lstm_cell_30/MatMulMatMul.sequential_13/lstm_30/strided_slice_2:output:0@sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_30/lstm_cell_30/MatMul?
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?
+sequential_13/lstm_30/lstm_cell_30/MatMul_1MatMul$sequential_13/lstm_30/zeros:output:0Bsequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_13/lstm_30/lstm_cell_30/MatMul_1?
&sequential_13/lstm_30/lstm_cell_30/addAddV23sequential_13/lstm_30/lstm_cell_30/MatMul:product:05sequential_13/lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_30/lstm_cell_30/add?
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?
*sequential_13/lstm_30/lstm_cell_30/BiasAddBiasAdd*sequential_13/lstm_30/lstm_cell_30/add:z:0Asequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_30/lstm_cell_30/BiasAdd?
2sequential_13/lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_30/lstm_cell_30/split/split_dim?
(sequential_13/lstm_30/lstm_cell_30/splitSplit;sequential_13/lstm_30/lstm_cell_30/split/split_dim:output:03sequential_13/lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2*
(sequential_13/lstm_30/lstm_cell_30/split?
*sequential_13/lstm_30/lstm_cell_30/SigmoidSigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_30/lstm_cell_30/Sigmoid?
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_1Sigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_1?
&sequential_13/lstm_30/lstm_cell_30/mulMul0sequential_13/lstm_30/lstm_cell_30/Sigmoid_1:y:0&sequential_13/lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_30/lstm_cell_30/mul?
'sequential_13/lstm_30/lstm_cell_30/ReluRelu1sequential_13/lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2)
'sequential_13/lstm_30/lstm_cell_30/Relu?
(sequential_13/lstm_30/lstm_cell_30/mul_1Mul.sequential_13/lstm_30/lstm_cell_30/Sigmoid:y:05sequential_13/lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_30/lstm_cell_30/mul_1?
(sequential_13/lstm_30/lstm_cell_30/add_1AddV2*sequential_13/lstm_30/lstm_cell_30/mul:z:0,sequential_13/lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_30/lstm_cell_30/add_1?
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_2Sigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_2?
)sequential_13/lstm_30/lstm_cell_30/Relu_1Relu,sequential_13/lstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_30/lstm_cell_30/Relu_1?
(sequential_13/lstm_30/lstm_cell_30/mul_2Mul0sequential_13/lstm_30/lstm_cell_30/Sigmoid_2:y:07sequential_13/lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_30/lstm_cell_30/mul_2?
3sequential_13/lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   25
3sequential_13/lstm_30/TensorArrayV2_1/element_shape?
%sequential_13/lstm_30/TensorArrayV2_1TensorListReserve<sequential_13/lstm_30/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_30/TensorArrayV2_1z
sequential_13/lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_30/time?
.sequential_13/lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_13/lstm_30/while/maximum_iterations?
(sequential_13/lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_30/while/loop_counter?
sequential_13/lstm_30/whileWhile1sequential_13/lstm_30/while/loop_counter:output:07sequential_13/lstm_30/while/maximum_iterations:output:0#sequential_13/lstm_30/time:output:0.sequential_13/lstm_30/TensorArrayV2_1:handle:0$sequential_13/lstm_30/zeros:output:0&sequential_13/lstm_30/zeros_1:output:0.sequential_13/lstm_30/strided_slice_1:output:0Msequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resourceCsequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resourceBsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_30_while_body_1254492*4
cond,R*
(sequential_13_lstm_30_while_cond_1254491*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_13/lstm_30/while?
Fsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2H
Fsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential_13/lstm_30/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_30/while:output:3Osequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02:
8sequential_13/lstm_30/TensorArrayV2Stack/TensorListStack?
+sequential_13/lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential_13/lstm_30/strided_slice_3/stack?
-sequential_13/lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_30/strided_slice_3/stack_1?
-sequential_13/lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_3/stack_2?
%sequential_13/lstm_30/strided_slice_3StridedSliceAsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_30/strided_slice_3/stack:output:06sequential_13/lstm_30/strided_slice_3/stack_1:output:06sequential_13/lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_3?
&sequential_13/lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_30/transpose_1/perm?
!sequential_13/lstm_30/transpose_1	TransposeAsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/lstm_30/transpose_1?
sequential_13/lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_30/runtime?
!sequential_13/dropout_48/IdentityIdentity%sequential_13/lstm_30/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/dropout_48/Identity?
sequential_13/lstm_31/ShapeShape*sequential_13/dropout_48/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_31/Shape?
)sequential_13/lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_31/strided_slice/stack?
+sequential_13/lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_31/strided_slice/stack_1?
+sequential_13/lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_31/strided_slice/stack_2?
#sequential_13/lstm_31/strided_sliceStridedSlice$sequential_13/lstm_31/Shape:output:02sequential_13/lstm_31/strided_slice/stack:output:04sequential_13/lstm_31/strided_slice/stack_1:output:04sequential_13/lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_31/strided_slice?
$sequential_13/lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_13/lstm_31/zeros/packed/1?
"sequential_13/lstm_31/zeros/packedPack,sequential_13/lstm_31/strided_slice:output:0-sequential_13/lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_31/zeros/packed?
!sequential_13/lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_31/zeros/Const?
sequential_13/lstm_31/zerosFill+sequential_13/lstm_31/zeros/packed:output:0*sequential_13/lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_31/zeros?
&sequential_13/lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_13/lstm_31/zeros_1/packed/1?
$sequential_13/lstm_31/zeros_1/packedPack,sequential_13/lstm_31/strided_slice:output:0/sequential_13/lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_31/zeros_1/packed?
#sequential_13/lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_31/zeros_1/Const?
sequential_13/lstm_31/zeros_1Fill-sequential_13/lstm_31/zeros_1/packed:output:0,sequential_13/lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_13/lstm_31/zeros_1?
$sequential_13/lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_31/transpose/perm?
sequential_13/lstm_31/transpose	Transpose*sequential_13/dropout_48/Identity:output:0-sequential_13/lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2!
sequential_13/lstm_31/transpose?
sequential_13/lstm_31/Shape_1Shape#sequential_13/lstm_31/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_31/Shape_1?
+sequential_13/lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_31/strided_slice_1/stack?
-sequential_13/lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_1/stack_1?
-sequential_13/lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_1/stack_2?
%sequential_13/lstm_31/strided_slice_1StridedSlice&sequential_13/lstm_31/Shape_1:output:04sequential_13/lstm_31/strided_slice_1/stack:output:06sequential_13/lstm_31/strided_slice_1/stack_1:output:06sequential_13/lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_1?
1sequential_13/lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_13/lstm_31/TensorArrayV2/element_shape?
#sequential_13/lstm_31/TensorArrayV2TensorListReserve:sequential_13/lstm_31/TensorArrayV2/element_shape:output:0.sequential_13/lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_31/TensorArrayV2?
Ksequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_31/transpose:y:0Tsequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor?
+sequential_13/lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_31/strided_slice_2/stack?
-sequential_13/lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_2/stack_1?
-sequential_13/lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_2/stack_2?
%sequential_13/lstm_31/strided_slice_2StridedSlice#sequential_13/lstm_31/transpose:y:04sequential_13/lstm_31/strided_slice_2/stack:output:06sequential_13/lstm_31/strided_slice_2/stack_1:output:06sequential_13/lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_2?
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp?
)sequential_13/lstm_31/lstm_cell_31/MatMulMatMul.sequential_13/lstm_31/strided_slice_2:output:0@sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_31/lstm_cell_31/MatMul?
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?
+sequential_13/lstm_31/lstm_cell_31/MatMul_1MatMul$sequential_13/lstm_31/zeros:output:0Bsequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_13/lstm_31/lstm_cell_31/MatMul_1?
&sequential_13/lstm_31/lstm_cell_31/addAddV23sequential_13/lstm_31/lstm_cell_31/MatMul:product:05sequential_13/lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_31/lstm_cell_31/add?
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?
*sequential_13/lstm_31/lstm_cell_31/BiasAddBiasAdd*sequential_13/lstm_31/lstm_cell_31/add:z:0Asequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_31/lstm_cell_31/BiasAdd?
2sequential_13/lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_31/lstm_cell_31/split/split_dim?
(sequential_13/lstm_31/lstm_cell_31/splitSplit;sequential_13/lstm_31/lstm_cell_31/split/split_dim:output:03sequential_13/lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2*
(sequential_13/lstm_31/lstm_cell_31/split?
*sequential_13/lstm_31/lstm_cell_31/SigmoidSigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_13/lstm_31/lstm_cell_31/Sigmoid?
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_1Sigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_1?
&sequential_13/lstm_31/lstm_cell_31/mulMul0sequential_13/lstm_31/lstm_cell_31/Sigmoid_1:y:0&sequential_13/lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_31/lstm_cell_31/mul?
'sequential_13/lstm_31/lstm_cell_31/ReluRelu1sequential_13/lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2)
'sequential_13/lstm_31/lstm_cell_31/Relu?
(sequential_13/lstm_31/lstm_cell_31/mul_1Mul.sequential_13/lstm_31/lstm_cell_31/Sigmoid:y:05sequential_13/lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_31/lstm_cell_31/mul_1?
(sequential_13/lstm_31/lstm_cell_31/add_1AddV2*sequential_13/lstm_31/lstm_cell_31/mul:z:0,sequential_13/lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_31/lstm_cell_31/add_1?
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_2Sigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_2?
)sequential_13/lstm_31/lstm_cell_31/Relu_1Relu,sequential_13/lstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)sequential_13/lstm_31/lstm_cell_31/Relu_1?
(sequential_13/lstm_31/lstm_cell_31/mul_2Mul0sequential_13/lstm_31/lstm_cell_31/Sigmoid_2:y:07sequential_13/lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2*
(sequential_13/lstm_31/lstm_cell_31/mul_2?
3sequential_13/lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   25
3sequential_13/lstm_31/TensorArrayV2_1/element_shape?
%sequential_13/lstm_31/TensorArrayV2_1TensorListReserve<sequential_13/lstm_31/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_31/TensorArrayV2_1z
sequential_13/lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_31/time?
.sequential_13/lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_13/lstm_31/while/maximum_iterations?
(sequential_13/lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_31/while/loop_counter?
sequential_13/lstm_31/whileWhile1sequential_13/lstm_31/while/loop_counter:output:07sequential_13/lstm_31/while/maximum_iterations:output:0#sequential_13/lstm_31/time:output:0.sequential_13/lstm_31/TensorArrayV2_1:handle:0$sequential_13/lstm_31/zeros:output:0&sequential_13/lstm_31/zeros_1:output:0.sequential_13/lstm_31/strided_slice_1:output:0Msequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resourceCsequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resourceBsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_31_while_body_1254632*4
cond,R*
(sequential_13_lstm_31_while_cond_1254631*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_13/lstm_31/while?
Fsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2H
Fsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential_13/lstm_31/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_31/while:output:3Osequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02:
8sequential_13/lstm_31/TensorArrayV2Stack/TensorListStack?
+sequential_13/lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential_13/lstm_31/strided_slice_3/stack?
-sequential_13/lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_31/strided_slice_3/stack_1?
-sequential_13/lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_3/stack_2?
%sequential_13/lstm_31/strided_slice_3StridedSliceAsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_31/strided_slice_3/stack:output:06sequential_13/lstm_31/strided_slice_3/stack_1:output:06sequential_13/lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_3?
&sequential_13/lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_31/transpose_1/perm?
!sequential_13/lstm_31/transpose_1	TransposeAsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/lstm_31/transpose_1?
sequential_13/lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_31/runtime?
!sequential_13/dropout_49/IdentityIdentity%sequential_13/lstm_31/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/dropout_49/Identity?
/sequential_13/dense_31/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_13/dense_31/Tensordot/ReadVariableOp?
%sequential_13/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_31/Tensordot/axes?
%sequential_13/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_31/Tensordot/free?
&sequential_13/dense_31/Tensordot/ShapeShape*sequential_13/dropout_49/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_31/Tensordot/Shape?
.sequential_13/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_31/Tensordot/GatherV2/axis?
)sequential_13/dense_31/Tensordot/GatherV2GatherV2/sequential_13/dense_31/Tensordot/Shape:output:0.sequential_13/dense_31/Tensordot/free:output:07sequential_13/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_31/Tensordot/GatherV2?
0sequential_13/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_31/Tensordot/GatherV2_1/axis?
+sequential_13/dense_31/Tensordot/GatherV2_1GatherV2/sequential_13/dense_31/Tensordot/Shape:output:0.sequential_13/dense_31/Tensordot/axes:output:09sequential_13/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_31/Tensordot/GatherV2_1?
&sequential_13/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_31/Tensordot/Const?
%sequential_13/dense_31/Tensordot/ProdProd2sequential_13/dense_31/Tensordot/GatherV2:output:0/sequential_13/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_31/Tensordot/Prod?
(sequential_13/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_31/Tensordot/Const_1?
'sequential_13/dense_31/Tensordot/Prod_1Prod4sequential_13/dense_31/Tensordot/GatherV2_1:output:01sequential_13/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_31/Tensordot/Prod_1?
,sequential_13/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_31/Tensordot/concat/axis?
'sequential_13/dense_31/Tensordot/concatConcatV2.sequential_13/dense_31/Tensordot/free:output:0.sequential_13/dense_31/Tensordot/axes:output:05sequential_13/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_31/Tensordot/concat?
&sequential_13/dense_31/Tensordot/stackPack.sequential_13/dense_31/Tensordot/Prod:output:00sequential_13/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_31/Tensordot/stack?
*sequential_13/dense_31/Tensordot/transpose	Transpose*sequential_13/dropout_49/Identity:output:00sequential_13/dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_13/dense_31/Tensordot/transpose?
(sequential_13/dense_31/Tensordot/ReshapeReshape.sequential_13/dense_31/Tensordot/transpose:y:0/sequential_13/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_13/dense_31/Tensordot/Reshape?
'sequential_13/dense_31/Tensordot/MatMulMatMul1sequential_13/dense_31/Tensordot/Reshape:output:07sequential_13/dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_13/dense_31/Tensordot/MatMul?
(sequential_13/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_13/dense_31/Tensordot/Const_2?
.sequential_13/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_31/Tensordot/concat_1/axis?
)sequential_13/dense_31/Tensordot/concat_1ConcatV22sequential_13/dense_31/Tensordot/GatherV2:output:01sequential_13/dense_31/Tensordot/Const_2:output:07sequential_13/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_31/Tensordot/concat_1?
 sequential_13/dense_31/TensordotReshape1sequential_13/dense_31/Tensordot/MatMul:product:02sequential_13/dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_13/dense_31/Tensordot?
-sequential_13/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_13/dense_31/BiasAdd/ReadVariableOp?
sequential_13/dense_31/BiasAddBiasAdd)sequential_13/dense_31/Tensordot:output:05sequential_13/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2 
sequential_13/dense_31/BiasAdd?
sequential_13/dense_31/ReluRelu'sequential_13/dense_31/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_13/dense_31/Relu?
!sequential_13/dropout_50/IdentityIdentity)sequential_13/dense_31/Relu:activations:0*
T0*,
_output_shapes
:??????????2#
!sequential_13/dropout_50/Identity?
/sequential_13/dense_32/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_32_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_13/dense_32/Tensordot/ReadVariableOp?
%sequential_13/dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_32/Tensordot/axes?
%sequential_13/dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_32/Tensordot/free?
&sequential_13/dense_32/Tensordot/ShapeShape*sequential_13/dropout_50/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_32/Tensordot/Shape?
.sequential_13/dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_32/Tensordot/GatherV2/axis?
)sequential_13/dense_32/Tensordot/GatherV2GatherV2/sequential_13/dense_32/Tensordot/Shape:output:0.sequential_13/dense_32/Tensordot/free:output:07sequential_13/dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_32/Tensordot/GatherV2?
0sequential_13/dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_32/Tensordot/GatherV2_1/axis?
+sequential_13/dense_32/Tensordot/GatherV2_1GatherV2/sequential_13/dense_32/Tensordot/Shape:output:0.sequential_13/dense_32/Tensordot/axes:output:09sequential_13/dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_32/Tensordot/GatherV2_1?
&sequential_13/dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_32/Tensordot/Const?
%sequential_13/dense_32/Tensordot/ProdProd2sequential_13/dense_32/Tensordot/GatherV2:output:0/sequential_13/dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_32/Tensordot/Prod?
(sequential_13/dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_32/Tensordot/Const_1?
'sequential_13/dense_32/Tensordot/Prod_1Prod4sequential_13/dense_32/Tensordot/GatherV2_1:output:01sequential_13/dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_32/Tensordot/Prod_1?
,sequential_13/dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_32/Tensordot/concat/axis?
'sequential_13/dense_32/Tensordot/concatConcatV2.sequential_13/dense_32/Tensordot/free:output:0.sequential_13/dense_32/Tensordot/axes:output:05sequential_13/dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_32/Tensordot/concat?
&sequential_13/dense_32/Tensordot/stackPack.sequential_13/dense_32/Tensordot/Prod:output:00sequential_13/dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_32/Tensordot/stack?
*sequential_13/dense_32/Tensordot/transpose	Transpose*sequential_13/dropout_50/Identity:output:00sequential_13/dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_13/dense_32/Tensordot/transpose?
(sequential_13/dense_32/Tensordot/ReshapeReshape.sequential_13/dense_32/Tensordot/transpose:y:0/sequential_13/dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_13/dense_32/Tensordot/Reshape?
'sequential_13/dense_32/Tensordot/MatMulMatMul1sequential_13/dense_32/Tensordot/Reshape:output:07sequential_13/dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_13/dense_32/Tensordot/MatMul?
(sequential_13/dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_13/dense_32/Tensordot/Const_2?
.sequential_13/dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_32/Tensordot/concat_1/axis?
)sequential_13/dense_32/Tensordot/concat_1ConcatV22sequential_13/dense_32/Tensordot/GatherV2:output:01sequential_13/dense_32/Tensordot/Const_2:output:07sequential_13/dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_32/Tensordot/concat_1?
 sequential_13/dense_32/TensordotReshape1sequential_13/dense_32/Tensordot/MatMul:product:02sequential_13/dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_13/dense_32/Tensordot?
-sequential_13/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_13/dense_32/BiasAdd/ReadVariableOp?
sequential_13/dense_32/BiasAddBiasAdd)sequential_13/dense_32/Tensordot:output:05sequential_13/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 
sequential_13/dense_32/BiasAdd?
IdentityIdentity'sequential_13/dense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp.^sequential_13/dense_31/BiasAdd/ReadVariableOp0^sequential_13/dense_31/Tensordot/ReadVariableOp.^sequential_13/dense_32/BiasAdd/ReadVariableOp0^sequential_13/dense_32/Tensordot/ReadVariableOp:^sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp9^sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp;^sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^sequential_13/lstm_29/while:^sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp9^sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp;^sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^sequential_13/lstm_30/while:^sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp9^sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp;^sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^sequential_13/lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2^
-sequential_13/dense_31/BiasAdd/ReadVariableOp-sequential_13/dense_31/BiasAdd/ReadVariableOp2b
/sequential_13/dense_31/Tensordot/ReadVariableOp/sequential_13/dense_31/Tensordot/ReadVariableOp2^
-sequential_13/dense_32/BiasAdd/ReadVariableOp-sequential_13/dense_32/BiasAdd/ReadVariableOp2b
/sequential_13/dense_32/Tensordot/ReadVariableOp/sequential_13/dense_32/Tensordot/ReadVariableOp2v
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp2x
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2:
sequential_13/lstm_29/whilesequential_13/lstm_29/while2v
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp2x
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2:
sequential_13/lstm_30/whilesequential_13/lstm_30/while2v
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp2x
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2:
sequential_13/lstm_31/whilesequential_13/lstm_31/while:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?
?
while_cond_1256786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1256786___redundant_placeholder05
1while_while_cond_1256786___redundant_placeholder15
1while_while_cond_1256786___redundant_placeholder25
1while_while_cond_1256786___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
lstm_31_while_cond_1258901,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3.
*lstm_31_while_less_lstm_31_strided_slice_1E
Alstm_31_while_lstm_31_while_cond_1258901___redundant_placeholder0E
Alstm_31_while_lstm_31_while_cond_1258901___redundant_placeholder1E
Alstm_31_while_lstm_31_while_cond_1258901___redundant_placeholder2E
Alstm_31_while_lstm_31_while_cond_1258901___redundant_placeholder3
lstm_31_while_identity
?
lstm_31/while/LessLesslstm_31_while_placeholder*lstm_31_while_less_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2
lstm_31/while/Lessu
lstm_31/while/IdentityIdentitylstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_31/while/Identity"9
lstm_31_while_identitylstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_sequential_13_layer_call_fn_1258040

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:
??
	unknown_7:	?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_12571242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1255582

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_1260872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260872___redundant_placeholder05
1while_while_cond_1260872___redundant_placeholder15
1while_while_cond_1260872___redundant_placeholder25
1while_while_cond_1260872___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1256047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1256047___redundant_placeholder05
1while_while_cond_1256047___redundant_placeholder15
1while_while_cond_1256047___redundant_placeholder25
1while_while_cond_1256047___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
e
,__inference_dropout_47_layer_call_fn_1259681

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12575922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261254

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?h
?
 __inference__traced_save_1261551
file_prefix.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_29_lstm_cell_29_kernel_read_readvariableopD
@savev2_lstm_29_lstm_cell_29_recurrent_kernel_read_readvariableop8
4savev2_lstm_29_lstm_cell_29_bias_read_readvariableop:
6savev2_lstm_30_lstm_cell_30_kernel_read_readvariableopD
@savev2_lstm_30_lstm_cell_30_recurrent_kernel_read_readvariableop8
4savev2_lstm_30_lstm_cell_30_bias_read_readvariableop:
6savev2_lstm_31_lstm_cell_31_kernel_read_readvariableopD
@savev2_lstm_31_lstm_cell_31_recurrent_kernel_read_readvariableop8
4savev2_lstm_31_lstm_cell_31_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableopA
=savev2_adam_lstm_29_lstm_cell_29_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_29_lstm_cell_29_bias_m_read_readvariableopA
=savev2_adam_lstm_30_lstm_cell_30_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_30_lstm_cell_30_bias_m_read_readvariableopA
=savev2_adam_lstm_31_lstm_cell_31_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_31_lstm_cell_31_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableopA
=savev2_adam_lstm_29_lstm_cell_29_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_29_lstm_cell_29_bias_v_read_readvariableopA
=savev2_adam_lstm_30_lstm_cell_30_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_30_lstm_cell_30_bias_v_read_readvariableopA
=savev2_adam_lstm_31_lstm_cell_31_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_31_lstm_cell_31_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_29_lstm_cell_29_kernel_read_readvariableop@savev2_lstm_29_lstm_cell_29_recurrent_kernel_read_readvariableop4savev2_lstm_29_lstm_cell_29_bias_read_readvariableop6savev2_lstm_30_lstm_cell_30_kernel_read_readvariableop@savev2_lstm_30_lstm_cell_30_recurrent_kernel_read_readvariableop4savev2_lstm_30_lstm_cell_30_bias_read_readvariableop6savev2_lstm_31_lstm_cell_31_kernel_read_readvariableop@savev2_lstm_31_lstm_cell_31_recurrent_kernel_read_readvariableop4savev2_lstm_31_lstm_cell_31_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop=savev2_adam_lstm_29_lstm_cell_29_kernel_m_read_readvariableopGsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_29_lstm_cell_29_bias_m_read_readvariableop=savev2_adam_lstm_30_lstm_cell_30_kernel_m_read_readvariableopGsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_30_lstm_cell_30_bias_m_read_readvariableop=savev2_adam_lstm_31_lstm_cell_31_kernel_m_read_readvariableopGsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_31_lstm_cell_31_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop=savev2_adam_lstm_29_lstm_cell_29_kernel_v_read_readvariableopGsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_29_lstm_cell_29_bias_v_read_readvariableop=savev2_adam_lstm_30_lstm_cell_30_kernel_v_read_readvariableopGsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_30_lstm_cell_30_bias_v_read_readvariableop=savev2_adam_lstm_31_lstm_cell_31_kernel_v_read_readvariableopGsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_31_lstm_cell_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?:: : : : : :	?:
??:?:
??:
??:?:
??:
??:?: : : : :
??:?:	?::	?:
??:?:
??:
??:?:
??:
??:?:
??:?:	?::	?:
??:?:
??:
??:?:
??:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::%(!

_output_shapes
:	?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:&,"
 
_output_shapes
:
??:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:1

_output_shapes
: 
?
?
)__inference_lstm_30_layer_call_fn_1259731

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12568712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1259587
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1255651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1255651___redundant_placeholder05
1while_while_cond_1255651___redundant_placeholder15
1while_while_cond_1255651___redundant_placeholder25
1while_while_cond_1255651___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_1260087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_1257085

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1257290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1257290___redundant_placeholder05
1while_while_cond_1257290___redundant_placeholder15
1while_while_cond_1257290___redundant_placeholder25
1while_while_cond_1257290___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_1259301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260814

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260730*
condR
while_cond_1260729*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?J
?

lstm_29_while_body_1258608,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	?Q
=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??K
<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	?O
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
??I
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	???1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_29/while/TensorArrayV2Read/TensorListGetItem?
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_29/while/lstm_cell_29/MatMul?
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
#lstm_29/while/lstm_cell_29/MatMul_1MatMullstm_29_while_placeholder_2:lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_29/while/lstm_cell_29/MatMul_1?
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/MatMul:product:0-lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_29/while/lstm_cell_29/add?
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd"lstm_29/while/lstm_cell_29/add:z:09lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_29/while/lstm_cell_29/BiasAdd?
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_29/while/lstm_cell_29/split/split_dim?
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:0+lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_29/while/lstm_cell_29/split?
"lstm_29/while/lstm_cell_29/SigmoidSigmoid)lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_29/while/lstm_cell_29/Sigmoid?
$lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid)lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_29/while/lstm_cell_29/Sigmoid_1?
lstm_29/while/lstm_cell_29/mulMul(lstm_29/while/lstm_cell_29/Sigmoid_1:y:0lstm_29_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_29/while/lstm_cell_29/mul?
lstm_29/while/lstm_cell_29/ReluRelu)lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_29/while/lstm_cell_29/Relu?
 lstm_29/while/lstm_cell_29/mul_1Mul&lstm_29/while/lstm_cell_29/Sigmoid:y:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/mul_1?
 lstm_29/while/lstm_cell_29/add_1AddV2"lstm_29/while/lstm_cell_29/mul:z:0$lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/add_1?
$lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid)lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_29/while/lstm_cell_29/Sigmoid_2?
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_29/while/lstm_cell_29/Relu_1?
 lstm_29/while/lstm_cell_29/mul_2Mul(lstm_29/while/lstm_cell_29/Sigmoid_2:y:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/mul_2?
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_29/while/TensorArrayV2Write/TensorListSetIteml
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add/y?
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/addp
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add_1/y?
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/add_1?
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity?
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_1?
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_2?
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_3?
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_2:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_29/while/Identity_4?
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_1:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_29/while/Identity_5?
lstm_29/while/NoOpNoOp2^lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1^lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp3^lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_29/while/NoOp"9
lstm_29_while_identitylstm_29/while/Identity:output:0"=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"z
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"|
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"x
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"?
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2d
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2h
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?V
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260671
inputs_0?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260587*
condR
while_cond_1260586*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?!
?
E__inference_dense_31_layer_call_and_return_conditional_losses_1261024

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259385
inputs_0>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259301*
condR
while_cond_1259300*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261384

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?V
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1259885
inputs_0?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259801*
condR
while_cond_1259800*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
.__inference_lstm_cell_31_layer_call_fn_1261303

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12560342
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1254984

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_1260086
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260086___redundant_placeholder05
1while_while_cond_1260086___redundant_placeholder15
1while_while_cond_1260086___redundant_placeholder25
1while_while_cond_1260086___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
%__inference_signature_wrapper_1258009
lstm_29_input
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:
??
	unknown_7:	?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_12547712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?
?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1254838

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?%
?
while_body_1254852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_1254876_0:	?0
while_lstm_cell_29_1254878_0:
??+
while_lstm_cell_29_1254880_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_1254876:	?.
while_lstm_cell_29_1254878:
??)
while_lstm_cell_29_1254880:	???*while/lstm_cell_29/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_1254876_0while_lstm_cell_29_1254878_0while_lstm_cell_29_1254880_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12548382,
*while/lstm_cell_29/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_29_1254876while_lstm_cell_29_1254876_0":
while_lstm_cell_29_1254878while_lstm_cell_29_1254878_0":
while_lstm_cell_29_1254880while_lstm_cell_29_1254880_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?^
?
(sequential_13_lstm_29_while_body_1254352H
Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counterN
Jsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations+
'sequential_13_lstm_29_while_placeholder-
)sequential_13_lstm_29_while_placeholder_1-
)sequential_13_lstm_29_while_placeholder_2-
)sequential_13_lstm_29_while_placeholder_3G
Csequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1_0?
sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	?_
Ksequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??Y
Jsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	?(
$sequential_13_lstm_29_while_identity*
&sequential_13_lstm_29_while_identity_1*
&sequential_13_lstm_29_while_identity_2*
&sequential_13_lstm_29_while_identity_3*
&sequential_13_lstm_29_while_identity_4*
&sequential_13_lstm_29_while_identity_5E
Asequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1?
}sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	?]
Isequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
??W
Hsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	????sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
Msequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Msequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_29_while_placeholderVsequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02A
?sequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem?
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02@
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?
/sequential_13/lstm_29/while/lstm_cell_29/MatMulMatMulFsequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_29/while/lstm_cell_29/MatMul?
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02B
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
1sequential_13/lstm_29/while/lstm_cell_29/MatMul_1MatMul)sequential_13_lstm_29_while_placeholder_2Hsequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_13/lstm_29/while/lstm_cell_29/MatMul_1?
,sequential_13/lstm_29/while/lstm_cell_29/addAddV29sequential_13/lstm_29/while/lstm_cell_29/MatMul:product:0;sequential_13/lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_29/while/lstm_cell_29/add?
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02A
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?
0sequential_13/lstm_29/while/lstm_cell_29/BiasAddBiasAdd0sequential_13/lstm_29/while/lstm_cell_29/add:z:0Gsequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_29/while/lstm_cell_29/BiasAdd?
8sequential_13/lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_29/while/lstm_cell_29/split/split_dim?
.sequential_13/lstm_29/while/lstm_cell_29/splitSplitAsequential_13/lstm_29/while/lstm_cell_29/split/split_dim:output:09sequential_13/lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split20
.sequential_13/lstm_29/while/lstm_cell_29/split?
0sequential_13/lstm_29/while/lstm_cell_29/SigmoidSigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_29/while/lstm_cell_29/Sigmoid?
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1?
,sequential_13/lstm_29/while/lstm_cell_29/mulMul6sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1:y:0)sequential_13_lstm_29_while_placeholder_3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_29/while/lstm_cell_29/mul?
-sequential_13/lstm_29/while/lstm_cell_29/ReluRelu7sequential_13/lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2/
-sequential_13/lstm_29/while/lstm_cell_29/Relu?
.sequential_13/lstm_29/while/lstm_cell_29/mul_1Mul4sequential_13/lstm_29/while/lstm_cell_29/Sigmoid:y:0;sequential_13/lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_29/while/lstm_cell_29/mul_1?
.sequential_13/lstm_29/while/lstm_cell_29/add_1AddV20sequential_13/lstm_29/while/lstm_cell_29/mul:z:02sequential_13/lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_29/while/lstm_cell_29/add_1?
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2?
/sequential_13/lstm_29/while/lstm_cell_29/Relu_1Relu2sequential_13/lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_29/while/lstm_cell_29/Relu_1?
.sequential_13/lstm_29/while/lstm_cell_29/mul_2Mul6sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2:y:0=sequential_13/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_29/while/lstm_cell_29/mul_2?
@sequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_29_while_placeholder_1'sequential_13_lstm_29_while_placeholder2sequential_13/lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItem?
!sequential_13/lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_29/while/add/y?
sequential_13/lstm_29/while/addAddV2'sequential_13_lstm_29_while_placeholder*sequential_13/lstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_29/while/add?
#sequential_13/lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_29/while/add_1/y?
!sequential_13/lstm_29/while/add_1AddV2Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counter,sequential_13/lstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_29/while/add_1?
$sequential_13/lstm_29/while/IdentityIdentity%sequential_13/lstm_29/while/add_1:z:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_29/while/Identity?
&sequential_13/lstm_29/while/Identity_1IdentityJsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_1?
&sequential_13/lstm_29/while/Identity_2Identity#sequential_13/lstm_29/while/add:z:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_2?
&sequential_13/lstm_29/while/Identity_3IdentityPsequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_3?
&sequential_13/lstm_29/while/Identity_4Identity2sequential_13/lstm_29/while/lstm_cell_29/mul_2:z:0!^sequential_13/lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_29/while/Identity_4?
&sequential_13/lstm_29/while/Identity_5Identity2sequential_13/lstm_29/while/lstm_cell_29/add_1:z:0!^sequential_13/lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_29/while/Identity_5?
 sequential_13/lstm_29/while/NoOpNoOp@^sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?^sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpA^sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_29/while/NoOp"U
$sequential_13_lstm_29_while_identity-sequential_13/lstm_29/while/Identity:output:0"Y
&sequential_13_lstm_29_while_identity_1/sequential_13/lstm_29/while/Identity_1:output:0"Y
&sequential_13_lstm_29_while_identity_2/sequential_13/lstm_29/while/Identity_2:output:0"Y
&sequential_13_lstm_29_while_identity_3/sequential_13/lstm_29/while/Identity_3:output:0"Y
&sequential_13_lstm_29_while_identity_4/sequential_13/lstm_29/while/Identity_4:output:0"Y
&sequential_13_lstm_29_while_identity_5/sequential_13/lstm_29/while/Identity_5:output:0"?
Hsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resourceJsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"?
Isequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resourceKsequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"?
Gsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resourceIsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"?
Asequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1Csequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1_0"?
}sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2?
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2?
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259242
inputs_0>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259158*
condR
while_cond_1259157*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1254851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1254851___redundant_placeholder05
1while_while_cond_1254851___redundant_placeholder15
1while_while_cond_1254851___redundant_placeholder25
1while_while_cond_1254851___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?J
?

lstm_30_while_body_1258270,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3+
'lstm_30_while_lstm_30_strided_slice_1_0g
clstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
??Q
=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??K
<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
lstm_30_while_identity
lstm_30_while_identity_1
lstm_30_while_identity_2
lstm_30_while_identity_3
lstm_30_while_identity_4
lstm_30_while_identity_5)
%lstm_30_while_lstm_30_strided_slice_1e
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorM
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
??O
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
??I
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	???1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0lstm_30_while_placeholderHlstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_30/while/TensorArrayV2Read/TensorListGetItem?
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?
!lstm_30/while/lstm_cell_30/MatMulMatMul8lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_30/while/lstm_cell_30/MatMul?
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
#lstm_30/while/lstm_cell_30/MatMul_1MatMullstm_30_while_placeholder_2:lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_30/while/lstm_cell_30/MatMul_1?
lstm_30/while/lstm_cell_30/addAddV2+lstm_30/while/lstm_cell_30/MatMul:product:0-lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_30/while/lstm_cell_30/add?
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?
"lstm_30/while/lstm_cell_30/BiasAddBiasAdd"lstm_30/while/lstm_cell_30/add:z:09lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_30/while/lstm_cell_30/BiasAdd?
*lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_30/while/lstm_cell_30/split/split_dim?
 lstm_30/while/lstm_cell_30/splitSplit3lstm_30/while/lstm_cell_30/split/split_dim:output:0+lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_30/while/lstm_cell_30/split?
"lstm_30/while/lstm_cell_30/SigmoidSigmoid)lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_30/while/lstm_cell_30/Sigmoid?
$lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid)lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_30/while/lstm_cell_30/Sigmoid_1?
lstm_30/while/lstm_cell_30/mulMul(lstm_30/while/lstm_cell_30/Sigmoid_1:y:0lstm_30_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_30/while/lstm_cell_30/mul?
lstm_30/while/lstm_cell_30/ReluRelu)lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_30/while/lstm_cell_30/Relu?
 lstm_30/while/lstm_cell_30/mul_1Mul&lstm_30/while/lstm_cell_30/Sigmoid:y:0-lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/mul_1?
 lstm_30/while/lstm_cell_30/add_1AddV2"lstm_30/while/lstm_cell_30/mul:z:0$lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/add_1?
$lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid)lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_30/while/lstm_cell_30/Sigmoid_2?
!lstm_30/while/lstm_cell_30/Relu_1Relu$lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_30/while/lstm_cell_30/Relu_1?
 lstm_30/while/lstm_cell_30/mul_2Mul(lstm_30/while/lstm_cell_30/Sigmoid_2:y:0/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/mul_2?
2lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_30_while_placeholder_1lstm_30_while_placeholder$lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_30/while/TensorArrayV2Write/TensorListSetIteml
lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add/y?
lstm_30/while/addAddV2lstm_30_while_placeholderlstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/addp
lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add_1/y?
lstm_30/while/add_1AddV2(lstm_30_while_lstm_30_while_loop_counterlstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/add_1?
lstm_30/while/IdentityIdentitylstm_30/while/add_1:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity?
lstm_30/while/Identity_1Identity.lstm_30_while_lstm_30_while_maximum_iterations^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_1?
lstm_30/while/Identity_2Identitylstm_30/while/add:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_2?
lstm_30/while/Identity_3IdentityBlstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_3?
lstm_30/while/Identity_4Identity$lstm_30/while/lstm_cell_30/mul_2:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_30/while/Identity_4?
lstm_30/while/Identity_5Identity$lstm_30/while/lstm_cell_30/add_1:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_30/while/Identity_5?
lstm_30/while/NoOpNoOp2^lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1^lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp3^lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_30/while/NoOp"9
lstm_30_while_identitylstm_30/while/Identity:output:0"=
lstm_30_while_identity_1!lstm_30/while/Identity_1:output:0"=
lstm_30_while_identity_2!lstm_30/while/Identity_2:output:0"=
lstm_30_while_identity_3!lstm_30/while/Identity_3:output:0"=
lstm_30_while_identity_4!lstm_30/while/Identity_4:output:0"=
lstm_30_while_identity_5!lstm_30/while/Identity_5:output:0"P
%lstm_30_while_lstm_30_strided_slice_1'lstm_30_while_lstm_30_strided_slice_1_0"z
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"|
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"x
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"?
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2d
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2h
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_lstm_29_layer_call_fn_1259099

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12577512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
lstm_30_while_cond_1258754,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3.
*lstm_30_while_less_lstm_30_strided_slice_1E
Alstm_30_while_lstm_30_while_cond_1258754___redundant_placeholder0E
Alstm_30_while_lstm_30_while_cond_1258754___redundant_placeholder1E
Alstm_30_while_lstm_30_while_cond_1258754___redundant_placeholder2E
Alstm_30_while_lstm_30_while_cond_1258754___redundant_placeholder3
lstm_30_while_identity
?
lstm_30/while/LessLesslstm_30_while_placeholder*lstm_30_while_less_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2
lstm_30/while/Lessu
lstm_30/while/IdentityIdentitylstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_30/while/Identity"9
lstm_30_while_identitylstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259528

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259444*
condR
while_cond_1259443*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_50_layer_call_fn_1261029

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12570852
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1260229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260229___redundant_placeholder05
1while_while_cond_1260229___redundant_placeholder15
1while_while_cond_1260229___redundant_placeholder25
1while_while_cond_1260229___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1259300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259300___redundant_placeholder05
1while_while_cond_1259300___redundant_placeholder15
1while_while_cond_1259300___redundant_placeholder25
1while_while_cond_1259300___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1259800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259800___redundant_placeholder05
1while_while_cond_1259800___redundant_placeholder15
1while_while_cond_1259800___redundant_placeholder25
1while_while_cond_1259800___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1255436

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_1256943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1256943___redundant_placeholder05
1while_while_cond_1256943___redundant_placeholder15
1while_while_cond_1256943___redundant_placeholder25
1while_while_cond_1256943___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_31_layer_call_fn_1260374

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12570282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_49_layer_call_fn_1260967

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12572162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_1257183

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?J
?

lstm_31_while_body_1258410,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3+
'lstm_31_while_lstm_31_strided_slice_1_0g
clstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
??Q
=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??K
<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
lstm_31_while_identity
lstm_31_while_identity_1
lstm_31_while_identity_2
lstm_31_while_identity_3
lstm_31_while_identity_4
lstm_31_while_identity_5)
%lstm_31_while_lstm_31_strided_slice_1e
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorM
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
??O
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
??I
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	???1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0lstm_31_while_placeholderHlstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_31/while/TensorArrayV2Read/TensorListGetItem?
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?
!lstm_31/while/lstm_cell_31/MatMulMatMul8lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_31/while/lstm_cell_31/MatMul?
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
#lstm_31/while/lstm_cell_31/MatMul_1MatMullstm_31_while_placeholder_2:lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_31/while/lstm_cell_31/MatMul_1?
lstm_31/while/lstm_cell_31/addAddV2+lstm_31/while/lstm_cell_31/MatMul:product:0-lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_31/while/lstm_cell_31/add?
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?
"lstm_31/while/lstm_cell_31/BiasAddBiasAdd"lstm_31/while/lstm_cell_31/add:z:09lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_31/while/lstm_cell_31/BiasAdd?
*lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_31/while/lstm_cell_31/split/split_dim?
 lstm_31/while/lstm_cell_31/splitSplit3lstm_31/while/lstm_cell_31/split/split_dim:output:0+lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_31/while/lstm_cell_31/split?
"lstm_31/while/lstm_cell_31/SigmoidSigmoid)lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_31/while/lstm_cell_31/Sigmoid?
$lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid)lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_31/while/lstm_cell_31/Sigmoid_1?
lstm_31/while/lstm_cell_31/mulMul(lstm_31/while/lstm_cell_31/Sigmoid_1:y:0lstm_31_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_31/while/lstm_cell_31/mul?
lstm_31/while/lstm_cell_31/ReluRelu)lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_31/while/lstm_cell_31/Relu?
 lstm_31/while/lstm_cell_31/mul_1Mul&lstm_31/while/lstm_cell_31/Sigmoid:y:0-lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/mul_1?
 lstm_31/while/lstm_cell_31/add_1AddV2"lstm_31/while/lstm_cell_31/mul:z:0$lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/add_1?
$lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid)lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_31/while/lstm_cell_31/Sigmoid_2?
!lstm_31/while/lstm_cell_31/Relu_1Relu$lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_31/while/lstm_cell_31/Relu_1?
 lstm_31/while/lstm_cell_31/mul_2Mul(lstm_31/while/lstm_cell_31/Sigmoid_2:y:0/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/mul_2?
2lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_31_while_placeholder_1lstm_31_while_placeholder$lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_31/while/TensorArrayV2Write/TensorListSetIteml
lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add/y?
lstm_31/while/addAddV2lstm_31_while_placeholderlstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/addp
lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add_1/y?
lstm_31/while/add_1AddV2(lstm_31_while_lstm_31_while_loop_counterlstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/add_1?
lstm_31/while/IdentityIdentitylstm_31/while/add_1:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity?
lstm_31/while/Identity_1Identity.lstm_31_while_lstm_31_while_maximum_iterations^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_1?
lstm_31/while/Identity_2Identitylstm_31/while/add:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_2?
lstm_31/while/Identity_3IdentityBlstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_3?
lstm_31/while/Identity_4Identity$lstm_31/while/lstm_cell_31/mul_2:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_31/while/Identity_4?
lstm_31/while/Identity_5Identity$lstm_31/while/lstm_cell_31/add_1:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_31/while/Identity_5?
lstm_31/while/NoOpNoOp2^lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1^lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp3^lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_31/while/NoOp"9
lstm_31_while_identitylstm_31/while/Identity:output:0"=
lstm_31_while_identity_1!lstm_31/while/Identity_1:output:0"=
lstm_31_while_identity_2!lstm_31/while/Identity_2:output:0"=
lstm_31_while_identity_3!lstm_31/while/Identity_3:output:0"=
lstm_31_while_identity_4!lstm_31/while/Identity_4:output:0"=
lstm_31_while_identity_5!lstm_31/while/Identity_5:output:0"P
%lstm_31_while_lstm_31_strided_slice_1'lstm_31_while_lstm_31_strided_slice_1_0"z
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"?
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2h
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1256871

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1256787*
condR
while_cond_1256786*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1258549

inputsF
3lstm_29_lstm_cell_29_matmul_readvariableop_resource:	?I
5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
??C
4lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	?G
3lstm_30_lstm_cell_30_matmul_readvariableop_resource:
??I
5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
??C
4lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	?G
3lstm_31_lstm_cell_31_matmul_readvariableop_resource:
??I
5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
??C
4lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	?>
*dense_31_tensordot_readvariableop_resource:
??7
(dense_31_biasadd_readvariableop_resource:	?=
*dense_32_tensordot_readvariableop_resource:	?6
(dense_32_biasadd_readvariableop_resource:
identity??dense_31/BiasAdd/ReadVariableOp?!dense_31/Tensordot/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?!dense_32/Tensordot/ReadVariableOp?+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?*lstm_29/lstm_cell_29/MatMul/ReadVariableOp?,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?lstm_29/while?+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?*lstm_30/lstm_cell_30/MatMul/ReadVariableOp?,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?lstm_30/while?+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?*lstm_31/lstm_cell_31/MatMul/ReadVariableOp?,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?lstm_31/whileT
lstm_29/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_29/Shape?
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice/stack?
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_1?
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_2?
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slices
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_29/zeros/packed/1?
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros/packedo
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros/Const?
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/zerosw
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_29/zeros_1/packed/1?
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros_1/packeds
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros_1/Const?
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/zeros_1?
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose/perm?
lstm_29/transpose	Transposeinputslstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_29/transposeg
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
:2
lstm_29/Shape_1?
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_1/stack?
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_1?
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_2?
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slice_1?
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_29/TensorArrayV2/element_shape?
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_29/TensorArrayUnstack/TensorListFromTensor?
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_2/stack?
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_1?
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_2?
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_29/strided_slice_2?
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp?
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:02lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/MatMul?
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_29/lstm_cell_29/MatMul_1MatMullstm_29/zeros:output:04lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/MatMul_1?
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/MatMul:product:0'lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/add?
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_29/lstm_cell_29/BiasAddBiasAddlstm_29/lstm_cell_29/add:z:03lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/BiasAdd?
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_29/lstm_cell_29/split/split_dim?
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:0%lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_29/lstm_cell_29/split?
lstm_29/lstm_cell_29/SigmoidSigmoid#lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Sigmoid?
lstm_29/lstm_cell_29/Sigmoid_1Sigmoid#lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_29/lstm_cell_29/Sigmoid_1?
lstm_29/lstm_cell_29/mulMul"lstm_29/lstm_cell_29/Sigmoid_1:y:0lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul?
lstm_29/lstm_cell_29/ReluRelu#lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Relu?
lstm_29/lstm_cell_29/mul_1Mul lstm_29/lstm_cell_29/Sigmoid:y:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul_1?
lstm_29/lstm_cell_29/add_1AddV2lstm_29/lstm_cell_29/mul:z:0lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/add_1?
lstm_29/lstm_cell_29/Sigmoid_2Sigmoid#lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_29/lstm_cell_29/Sigmoid_2?
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Relu_1?
lstm_29/lstm_cell_29/mul_2Mul"lstm_29/lstm_cell_29/Sigmoid_2:y:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul_2?
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_29/TensorArrayV2_1/element_shape?
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2_1^
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/time?
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_29/while/maximum_iterationsz
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/while/loop_counter?
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_29_lstm_cell_29_matmul_readvariableop_resource5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_29_while_body_1258130*&
condR
lstm_29_while_cond_1258129*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_29/while?
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_29/TensorArrayV2Stack/TensorListStack?
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_29/strided_slice_3/stack?
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_29/strided_slice_3/stack_1?
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_3/stack_2?
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_29/strided_slice_3?
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose_1/perm?
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_29/transpose_1v
lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/runtime?
dropout_47/IdentityIdentitylstm_29/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_47/Identityj
lstm_30/ShapeShapedropout_47/Identity:output:0*
T0*
_output_shapes
:2
lstm_30/Shape?
lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice/stack?
lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_1?
lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_2?
lstm_30/strided_sliceStridedSlicelstm_30/Shape:output:0$lstm_30/strided_slice/stack:output:0&lstm_30/strided_slice/stack_1:output:0&lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slices
lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_30/zeros/packed/1?
lstm_30/zeros/packedPacklstm_30/strided_slice:output:0lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros/packedo
lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros/Const?
lstm_30/zerosFilllstm_30/zeros/packed:output:0lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/zerosw
lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_30/zeros_1/packed/1?
lstm_30/zeros_1/packedPacklstm_30/strided_slice:output:0!lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros_1/packeds
lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros_1/Const?
lstm_30/zeros_1Filllstm_30/zeros_1/packed:output:0lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/zeros_1?
lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose/perm?
lstm_30/transpose	Transposedropout_47/Identity:output:0lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_30/transposeg
lstm_30/Shape_1Shapelstm_30/transpose:y:0*
T0*
_output_shapes
:2
lstm_30/Shape_1?
lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_1/stack?
lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_1?
lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_2?
lstm_30/strided_slice_1StridedSlicelstm_30/Shape_1:output:0&lstm_30/strided_slice_1/stack:output:0(lstm_30/strided_slice_1/stack_1:output:0(lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slice_1?
#lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_30/TensorArrayV2/element_shape?
lstm_30/TensorArrayV2TensorListReserve,lstm_30/TensorArrayV2/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_30/transpose:y:0Flstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_30/TensorArrayUnstack/TensorListFromTensor?
lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_2/stack?
lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_1?
lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_2?
lstm_30/strided_slice_2StridedSlicelstm_30/transpose:y:0&lstm_30/strided_slice_2/stack:output:0(lstm_30/strided_slice_2/stack_1:output:0(lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_30/strided_slice_2?
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp?
lstm_30/lstm_cell_30/MatMulMatMul lstm_30/strided_slice_2:output:02lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/MatMul?
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_30/lstm_cell_30/MatMul_1MatMullstm_30/zeros:output:04lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/MatMul_1?
lstm_30/lstm_cell_30/addAddV2%lstm_30/lstm_cell_30/MatMul:product:0'lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/add?
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_30/lstm_cell_30/BiasAddBiasAddlstm_30/lstm_cell_30/add:z:03lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/BiasAdd?
$lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_30/lstm_cell_30/split/split_dim?
lstm_30/lstm_cell_30/splitSplit-lstm_30/lstm_cell_30/split/split_dim:output:0%lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_30/lstm_cell_30/split?
lstm_30/lstm_cell_30/SigmoidSigmoid#lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Sigmoid?
lstm_30/lstm_cell_30/Sigmoid_1Sigmoid#lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_30/lstm_cell_30/Sigmoid_1?
lstm_30/lstm_cell_30/mulMul"lstm_30/lstm_cell_30/Sigmoid_1:y:0lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul?
lstm_30/lstm_cell_30/ReluRelu#lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Relu?
lstm_30/lstm_cell_30/mul_1Mul lstm_30/lstm_cell_30/Sigmoid:y:0'lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul_1?
lstm_30/lstm_cell_30/add_1AddV2lstm_30/lstm_cell_30/mul:z:0lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/add_1?
lstm_30/lstm_cell_30/Sigmoid_2Sigmoid#lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_30/lstm_cell_30/Sigmoid_2?
lstm_30/lstm_cell_30/Relu_1Relulstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Relu_1?
lstm_30/lstm_cell_30/mul_2Mul"lstm_30/lstm_cell_30/Sigmoid_2:y:0)lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul_2?
%lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_30/TensorArrayV2_1/element_shape?
lstm_30/TensorArrayV2_1TensorListReserve.lstm_30/TensorArrayV2_1/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2_1^
lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/time?
 lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_30/while/maximum_iterationsz
lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/while/loop_counter?
lstm_30/whileWhile#lstm_30/while/loop_counter:output:0)lstm_30/while/maximum_iterations:output:0lstm_30/time:output:0 lstm_30/TensorArrayV2_1:handle:0lstm_30/zeros:output:0lstm_30/zeros_1:output:0 lstm_30/strided_slice_1:output:0?lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_30_lstm_cell_30_matmul_readvariableop_resource5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_30_while_body_1258270*&
condR
lstm_30_while_cond_1258269*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_30/while?
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_30/TensorArrayV2Stack/TensorListStackTensorListStacklstm_30/while:output:3Alstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_30/TensorArrayV2Stack/TensorListStack?
lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_30/strided_slice_3/stack?
lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_30/strided_slice_3/stack_1?
lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_3/stack_2?
lstm_30/strided_slice_3StridedSlice3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_30/strided_slice_3/stack:output:0(lstm_30/strided_slice_3/stack_1:output:0(lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_30/strided_slice_3?
lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose_1/perm?
lstm_30/transpose_1	Transpose3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_30/transpose_1v
lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/runtime?
dropout_48/IdentityIdentitylstm_30/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_48/Identityj
lstm_31/ShapeShapedropout_48/Identity:output:0*
T0*
_output_shapes
:2
lstm_31/Shape?
lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice/stack?
lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_1?
lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_2?
lstm_31/strided_sliceStridedSlicelstm_31/Shape:output:0$lstm_31/strided_slice/stack:output:0&lstm_31/strided_slice/stack_1:output:0&lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slices
lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_31/zeros/packed/1?
lstm_31/zeros/packedPacklstm_31/strided_slice:output:0lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros/packedo
lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros/Const?
lstm_31/zerosFilllstm_31/zeros/packed:output:0lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/zerosw
lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_31/zeros_1/packed/1?
lstm_31/zeros_1/packedPacklstm_31/strided_slice:output:0!lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros_1/packeds
lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros_1/Const?
lstm_31/zeros_1Filllstm_31/zeros_1/packed:output:0lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/zeros_1?
lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose/perm?
lstm_31/transpose	Transposedropout_48/Identity:output:0lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_31/transposeg
lstm_31/Shape_1Shapelstm_31/transpose:y:0*
T0*
_output_shapes
:2
lstm_31/Shape_1?
lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_1/stack?
lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_1?
lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_2?
lstm_31/strided_slice_1StridedSlicelstm_31/Shape_1:output:0&lstm_31/strided_slice_1/stack:output:0(lstm_31/strided_slice_1/stack_1:output:0(lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slice_1?
#lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_31/TensorArrayV2/element_shape?
lstm_31/TensorArrayV2TensorListReserve,lstm_31/TensorArrayV2/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_31/transpose:y:0Flstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_31/TensorArrayUnstack/TensorListFromTensor?
lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_2/stack?
lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_1?
lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_2?
lstm_31/strided_slice_2StridedSlicelstm_31/transpose:y:0&lstm_31/strided_slice_2/stack:output:0(lstm_31/strided_slice_2/stack_1:output:0(lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_31/strided_slice_2?
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp?
lstm_31/lstm_cell_31/MatMulMatMul lstm_31/strided_slice_2:output:02lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/MatMul?
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_31/lstm_cell_31/MatMul_1MatMullstm_31/zeros:output:04lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/MatMul_1?
lstm_31/lstm_cell_31/addAddV2%lstm_31/lstm_cell_31/MatMul:product:0'lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/add?
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_31/lstm_cell_31/BiasAddBiasAddlstm_31/lstm_cell_31/add:z:03lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/BiasAdd?
$lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_31/lstm_cell_31/split/split_dim?
lstm_31/lstm_cell_31/splitSplit-lstm_31/lstm_cell_31/split/split_dim:output:0%lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_31/lstm_cell_31/split?
lstm_31/lstm_cell_31/SigmoidSigmoid#lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Sigmoid?
lstm_31/lstm_cell_31/Sigmoid_1Sigmoid#lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_31/lstm_cell_31/Sigmoid_1?
lstm_31/lstm_cell_31/mulMul"lstm_31/lstm_cell_31/Sigmoid_1:y:0lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul?
lstm_31/lstm_cell_31/ReluRelu#lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Relu?
lstm_31/lstm_cell_31/mul_1Mul lstm_31/lstm_cell_31/Sigmoid:y:0'lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul_1?
lstm_31/lstm_cell_31/add_1AddV2lstm_31/lstm_cell_31/mul:z:0lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/add_1?
lstm_31/lstm_cell_31/Sigmoid_2Sigmoid#lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_31/lstm_cell_31/Sigmoid_2?
lstm_31/lstm_cell_31/Relu_1Relulstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Relu_1?
lstm_31/lstm_cell_31/mul_2Mul"lstm_31/lstm_cell_31/Sigmoid_2:y:0)lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul_2?
%lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_31/TensorArrayV2_1/element_shape?
lstm_31/TensorArrayV2_1TensorListReserve.lstm_31/TensorArrayV2_1/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2_1^
lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/time?
 lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_31/while/maximum_iterationsz
lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/while/loop_counter?
lstm_31/whileWhile#lstm_31/while/loop_counter:output:0)lstm_31/while/maximum_iterations:output:0lstm_31/time:output:0 lstm_31/TensorArrayV2_1:handle:0lstm_31/zeros:output:0lstm_31/zeros_1:output:0 lstm_31/strided_slice_1:output:0?lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_31_lstm_cell_31_matmul_readvariableop_resource5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_31_while_body_1258410*&
condR
lstm_31_while_cond_1258409*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_31/while?
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_31/TensorArrayV2Stack/TensorListStackTensorListStacklstm_31/while:output:3Alstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_31/TensorArrayV2Stack/TensorListStack?
lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_31/strided_slice_3/stack?
lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_31/strided_slice_3/stack_1?
lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_3/stack_2?
lstm_31/strided_slice_3StridedSlice3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_31/strided_slice_3/stack:output:0(lstm_31/strided_slice_3/stack_1:output:0(lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_31/strided_slice_3?
lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose_1/perm?
lstm_31/transpose_1	Transpose3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_31/transpose_1v
lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/runtime?
dropout_49/IdentityIdentitylstm_31/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_49/Identity?
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes?
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free?
dense_31/Tensordot/ShapeShapedropout_49/Identity:output:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape?
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axis?
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2?
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis?
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const?
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod?
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1?
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1?
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axis?
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat?
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stack?
dense_31/Tensordot/transpose	Transposedropout_49/Identity:output:0"dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Tensordot/transpose?
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_31/Tensordot/Reshape?
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/Tensordot/MatMul?
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_31/Tensordot/Const_2?
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axis?
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1?
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Tensordot?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_31/BiasAddx
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Relu?
dropout_50/IdentityIdentitydense_31/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_50/Identity?
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes?
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/free?
dense_32/Tensordot/ShapeShapedropout_50/Identity:output:0*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape?
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axis?
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2?
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis?
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const?
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod?
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1?
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1?
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axis?
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat?
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stack?
dense_32/Tensordot/transpose	Transposedropout_50/Identity:output:0"dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_32/Tensordot/transpose?
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_32/Tensordot/Reshape?
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/Tensordot/MatMul?
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/Const_2?
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axis?
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1?
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_32/Tensordot?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_32/BiasAddx
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp"^dense_32/Tensordot/ReadVariableOp,^lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+^lstm_29/lstm_cell_29/MatMul/ReadVariableOp-^lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^lstm_29/while,^lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+^lstm_30/lstm_cell_30/MatMul/ReadVariableOp-^lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^lstm_30/while,^lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+^lstm_31/lstm_cell_31/MatMul/ReadVariableOp-^lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2F
!dense_32/Tensordot/ReadVariableOp!dense_32/Tensordot/ReadVariableOp2Z
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2X
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp*lstm_29/lstm_cell_29/MatMul/ReadVariableOp2\
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2
lstm_29/whilelstm_29/while2Z
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2X
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp*lstm_30/lstm_cell_30/MatMul/ReadVariableOp2\
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2
lstm_30/whilelstm_30/while2Z
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2X
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp*lstm_31/lstm_cell_31/MatMul/ReadVariableOp2\
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2
lstm_31/whilelstm_31/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1255519

inputs(
lstm_cell_30_1255437:
??(
lstm_cell_30_1255439:
??#
lstm_cell_30_1255441:	?
identity??$lstm_cell_30/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_1255437lstm_cell_30_1255439lstm_cell_30_1255441*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12554362&
$lstm_cell_30/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_1255437lstm_cell_30_1255439lstm_cell_30_1255441*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1255450*
condR
while_cond_1255449*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_1256884

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257931
lstm_29_input"
lstm_29_1257895:	?#
lstm_29_1257897:
??
lstm_29_1257899:	?#
lstm_30_1257903:
??#
lstm_30_1257905:
??
lstm_30_1257907:	?#
lstm_31_1257911:
??#
lstm_31_1257913:
??
lstm_31_1257915:	?$
dense_31_1257919:
??
dense_31_1257921:	?#
dense_32_1257925:	?
dense_32_1257927:
identity?? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall?lstm_29/StatefulPartitionedCall?lstm_30/StatefulPartitionedCall?lstm_31/StatefulPartitionedCall?
lstm_29/StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputlstm_29_1257895lstm_29_1257897lstm_29_1257899*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12567142!
lstm_29/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12567272
dropout_47/PartitionedCall?
lstm_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0lstm_30_1257903lstm_30_1257905lstm_30_1257907*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12568712!
lstm_30/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall(lstm_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12568842
dropout_48/PartitionedCall?
lstm_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0lstm_31_1257911lstm_31_1257913lstm_31_1257915*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12570282!
lstm_31/StatefulPartitionedCall?
dropout_49/PartitionedCallPartitionedCall(lstm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12570412
dropout_49/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_31_1257919dense_31_1257921*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_12570742"
 dense_31/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12570852
dropout_50/PartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_32_1257925dense_32_1257927*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_12571172"
 dense_32/StatefulPartitionedCall?
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?
?
while_cond_1255449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1255449___redundant_placeholder05
1while_while_cond_1255449___redundant_placeholder15
1while_while_cond_1255449___redundant_placeholder25
1while_while_cond_1255449___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1257666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1257666___redundant_placeholder05
1while_while_cond_1257666___redundant_placeholder15
1while_while_cond_1257666___redundant_placeholder25
1while_while_cond_1257666___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1256249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1256249___redundant_placeholder05
1while_while_cond_1256249___redundant_placeholder15
1while_while_cond_1256249___redundant_placeholder25
1while_while_cond_1256249___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
H
,__inference_dropout_49_layer_call_fn_1260962

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12570412
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_47_layer_call_fn_1259676

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12567272
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_lstm_30_layer_call_fn_1259720
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12557212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?J
?

lstm_29_while_body_1258130,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	?Q
=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??K
<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	?O
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
??I
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	???1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_29/while/TensorArrayV2Read/TensorListGetItem?
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp?
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_29/while/lstm_cell_29/MatMul?
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp?
#lstm_29/while/lstm_cell_29/MatMul_1MatMullstm_29_while_placeholder_2:lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_29/while/lstm_cell_29/MatMul_1?
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/MatMul:product:0-lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_29/while/lstm_cell_29/add?
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd"lstm_29/while/lstm_cell_29/add:z:09lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_29/while/lstm_cell_29/BiasAdd?
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_29/while/lstm_cell_29/split/split_dim?
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:0+lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_29/while/lstm_cell_29/split?
"lstm_29/while/lstm_cell_29/SigmoidSigmoid)lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_29/while/lstm_cell_29/Sigmoid?
$lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid)lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_29/while/lstm_cell_29/Sigmoid_1?
lstm_29/while/lstm_cell_29/mulMul(lstm_29/while/lstm_cell_29/Sigmoid_1:y:0lstm_29_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_29/while/lstm_cell_29/mul?
lstm_29/while/lstm_cell_29/ReluRelu)lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_29/while/lstm_cell_29/Relu?
 lstm_29/while/lstm_cell_29/mul_1Mul&lstm_29/while/lstm_cell_29/Sigmoid:y:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/mul_1?
 lstm_29/while/lstm_cell_29/add_1AddV2"lstm_29/while/lstm_cell_29/mul:z:0$lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/add_1?
$lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid)lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_29/while/lstm_cell_29/Sigmoid_2?
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_29/while/lstm_cell_29/Relu_1?
 lstm_29/while/lstm_cell_29/mul_2Mul(lstm_29/while/lstm_cell_29/Sigmoid_2:y:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_29/while/lstm_cell_29/mul_2?
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_29/while/TensorArrayV2Write/TensorListSetIteml
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add/y?
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/addp
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add_1/y?
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/add_1?
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity?
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_1?
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_2?
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_3?
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_2:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_29/while/Identity_4?
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_1:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_29/while/Identity_5?
lstm_29/while/NoOpNoOp2^lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1^lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp3^lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_29/while/NoOp"9
lstm_29_while_identitylstm_29/while/Identity:output:0"=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"z
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"|
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"x
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"?
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2d
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2h
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_30_while_cond_1258269,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3.
*lstm_30_while_less_lstm_30_strided_slice_1E
Alstm_30_while_lstm_30_while_cond_1258269___redundant_placeholder0E
Alstm_30_while_lstm_30_while_cond_1258269___redundant_placeholder1E
Alstm_30_while_lstm_30_while_cond_1258269___redundant_placeholder2E
Alstm_30_while_lstm_30_while_cond_1258269___redundant_placeholder3
lstm_30_while_identity
?
lstm_30/while/LessLesslstm_30_while_placeholder*lstm_30_while_less_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2
lstm_30/while/Lessu
lstm_30/while/IdentityIdentitylstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_30/while/Identity"9
lstm_30_while_identitylstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_31_layer_call_fn_1260352
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12561172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1259943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259943___redundant_placeholder05
1while_while_cond_1259943___redundant_placeholder15
1while_while_cond_1259943___redundant_placeholder25
1while_while_cond_1259943___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
e
,__inference_dropout_48_layer_call_fn_1260324

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12574042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260028
inputs_0?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259944*
condR
while_cond_1259943*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?%
?
while_body_1256048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_31_1256072_0:
??0
while_lstm_cell_31_1256074_0:
??+
while_lstm_cell_31_1256076_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_31_1256072:
??.
while_lstm_cell_31_1256074:
??)
while_lstm_cell_31_1256076:	???*while/lstm_cell_31/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_1256072_0while_lstm_cell_31_1256074_0while_lstm_cell_31_1256076_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12560342,
*while/lstm_cell_31/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_31_1256072while_lstm_cell_31_1256072_0":
while_lstm_cell_31_1256074while_lstm_cell_31_1256074_0":
while_lstm_cell_31_1256076while_lstm_cell_31_1256076_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_lstm_cell_30_layer_call_fn_1261222

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12555822
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1256714

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1256630*
condR
while_cond_1256629*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_lstm_cell_31_layer_call_fn_1261320

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12561802
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_1257592

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1259944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1257751

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1257667*
condR
while_cond_1257666*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?J
?

lstm_31_while_body_1258902,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3+
'lstm_31_while_lstm_31_strided_slice_1_0g
clstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
??Q
=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??K
<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
lstm_31_while_identity
lstm_31_while_identity_1
lstm_31_while_identity_2
lstm_31_while_identity_3
lstm_31_while_identity_4
lstm_31_while_identity_5)
%lstm_31_while_lstm_31_strided_slice_1e
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorM
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
??O
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
??I
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	???1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0lstm_31_while_placeholderHlstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_31/while/TensorArrayV2Read/TensorListGetItem?
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?
!lstm_31/while/lstm_cell_31/MatMulMatMul8lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_31/while/lstm_cell_31/MatMul?
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
#lstm_31/while/lstm_cell_31/MatMul_1MatMullstm_31_while_placeholder_2:lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_31/while/lstm_cell_31/MatMul_1?
lstm_31/while/lstm_cell_31/addAddV2+lstm_31/while/lstm_cell_31/MatMul:product:0-lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_31/while/lstm_cell_31/add?
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?
"lstm_31/while/lstm_cell_31/BiasAddBiasAdd"lstm_31/while/lstm_cell_31/add:z:09lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_31/while/lstm_cell_31/BiasAdd?
*lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_31/while/lstm_cell_31/split/split_dim?
 lstm_31/while/lstm_cell_31/splitSplit3lstm_31/while/lstm_cell_31/split/split_dim:output:0+lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_31/while/lstm_cell_31/split?
"lstm_31/while/lstm_cell_31/SigmoidSigmoid)lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_31/while/lstm_cell_31/Sigmoid?
$lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid)lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_31/while/lstm_cell_31/Sigmoid_1?
lstm_31/while/lstm_cell_31/mulMul(lstm_31/while/lstm_cell_31/Sigmoid_1:y:0lstm_31_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_31/while/lstm_cell_31/mul?
lstm_31/while/lstm_cell_31/ReluRelu)lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_31/while/lstm_cell_31/Relu?
 lstm_31/while/lstm_cell_31/mul_1Mul&lstm_31/while/lstm_cell_31/Sigmoid:y:0-lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/mul_1?
 lstm_31/while/lstm_cell_31/add_1AddV2"lstm_31/while/lstm_cell_31/mul:z:0$lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/add_1?
$lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid)lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_31/while/lstm_cell_31/Sigmoid_2?
!lstm_31/while/lstm_cell_31/Relu_1Relu$lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_31/while/lstm_cell_31/Relu_1?
 lstm_31/while/lstm_cell_31/mul_2Mul(lstm_31/while/lstm_cell_31/Sigmoid_2:y:0/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_31/while/lstm_cell_31/mul_2?
2lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_31_while_placeholder_1lstm_31_while_placeholder$lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_31/while/TensorArrayV2Write/TensorListSetIteml
lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add/y?
lstm_31/while/addAddV2lstm_31_while_placeholderlstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/addp
lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add_1/y?
lstm_31/while/add_1AddV2(lstm_31_while_lstm_31_while_loop_counterlstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/add_1?
lstm_31/while/IdentityIdentitylstm_31/while/add_1:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity?
lstm_31/while/Identity_1Identity.lstm_31_while_lstm_31_while_maximum_iterations^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_1?
lstm_31/while/Identity_2Identitylstm_31/while/add:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_2?
lstm_31/while/Identity_3IdentityBlstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_3?
lstm_31/while/Identity_4Identity$lstm_31/while/lstm_cell_31/mul_2:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_31/while/Identity_4?
lstm_31/while/Identity_5Identity$lstm_31/while/lstm_cell_31/add_1:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_31/while/Identity_5?
lstm_31/while/NoOpNoOp2^lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1^lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp3^lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_31/while/NoOp"9
lstm_31_while_identitylstm_31/while/Identity:output:0"=
lstm_31_while_identity_1!lstm_31/while/Identity_1:output:0"=
lstm_31_while_identity_2!lstm_31/while/Identity_2:output:0"=
lstm_31_while_identity_3!lstm_31/while/Identity_3:output:0"=
lstm_31_while_identity_4!lstm_31/while/Identity_4:output:0"=
lstm_31_while_identity_5!lstm_31/while/Identity_5:output:0"P
%lstm_31_while_lstm_31_strided_slice_1'lstm_31_while_lstm_31_strided_slice_1_0"z
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"?
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2h
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
? 
?
E__inference_dense_32_layer_call_and_return_conditional_losses_1261090

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1260587
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_48_layer_call_fn_1260319

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12568842
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1260730
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261352

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
)__inference_lstm_30_layer_call_fn_1259742

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12575632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_lstm_31_layer_call_fn_1260385

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12573752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_1257404

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1259444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?J
?

lstm_30_while_body_1258755,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3+
'lstm_30_while_lstm_30_strided_slice_1_0g
clstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
??Q
=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??K
<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
lstm_30_while_identity
lstm_30_while_identity_1
lstm_30_while_identity_2
lstm_30_while_identity_3
lstm_30_while_identity_4
lstm_30_while_identity_5)
%lstm_30_while_lstm_30_strided_slice_1e
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorM
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
??O
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
??I
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	???1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0lstm_30_while_placeholderHlstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_30/while/TensorArrayV2Read/TensorListGetItem?
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp?
!lstm_30/while/lstm_cell_30/MatMulMatMul8lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_30/while/lstm_cell_30/MatMul?
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp?
#lstm_30/while/lstm_cell_30/MatMul_1MatMullstm_30_while_placeholder_2:lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_30/while/lstm_cell_30/MatMul_1?
lstm_30/while/lstm_cell_30/addAddV2+lstm_30/while/lstm_cell_30/MatMul:product:0-lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_30/while/lstm_cell_30/add?
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?
"lstm_30/while/lstm_cell_30/BiasAddBiasAdd"lstm_30/while/lstm_cell_30/add:z:09lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_30/while/lstm_cell_30/BiasAdd?
*lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_30/while/lstm_cell_30/split/split_dim?
 lstm_30/while/lstm_cell_30/splitSplit3lstm_30/while/lstm_cell_30/split/split_dim:output:0+lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_30/while/lstm_cell_30/split?
"lstm_30/while/lstm_cell_30/SigmoidSigmoid)lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_30/while/lstm_cell_30/Sigmoid?
$lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid)lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_30/while/lstm_cell_30/Sigmoid_1?
lstm_30/while/lstm_cell_30/mulMul(lstm_30/while/lstm_cell_30/Sigmoid_1:y:0lstm_30_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_30/while/lstm_cell_30/mul?
lstm_30/while/lstm_cell_30/ReluRelu)lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_30/while/lstm_cell_30/Relu?
 lstm_30/while/lstm_cell_30/mul_1Mul&lstm_30/while/lstm_cell_30/Sigmoid:y:0-lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/mul_1?
 lstm_30/while/lstm_cell_30/add_1AddV2"lstm_30/while/lstm_cell_30/mul:z:0$lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/add_1?
$lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid)lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_30/while/lstm_cell_30/Sigmoid_2?
!lstm_30/while/lstm_cell_30/Relu_1Relu$lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_30/while/lstm_cell_30/Relu_1?
 lstm_30/while/lstm_cell_30/mul_2Mul(lstm_30/while/lstm_cell_30/Sigmoid_2:y:0/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_30/while/lstm_cell_30/mul_2?
2lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_30_while_placeholder_1lstm_30_while_placeholder$lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_30/while/TensorArrayV2Write/TensorListSetIteml
lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add/y?
lstm_30/while/addAddV2lstm_30_while_placeholderlstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/addp
lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add_1/y?
lstm_30/while/add_1AddV2(lstm_30_while_lstm_30_while_loop_counterlstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/add_1?
lstm_30/while/IdentityIdentitylstm_30/while/add_1:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity?
lstm_30/while/Identity_1Identity.lstm_30_while_lstm_30_while_maximum_iterations^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_1?
lstm_30/while/Identity_2Identitylstm_30/while/add:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_2?
lstm_30/while/Identity_3IdentityBlstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_3?
lstm_30/while/Identity_4Identity$lstm_30/while/lstm_cell_30/mul_2:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_30/while/Identity_4?
lstm_30/while/Identity_5Identity$lstm_30/while/lstm_cell_30/add_1:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_30/while/Identity_5?
lstm_30/while/NoOpNoOp2^lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1^lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp3^lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_30/while/NoOp"9
lstm_30_while_identitylstm_30/while/Identity:output:0"=
lstm_30_while_identity_1!lstm_30/while/Identity_1:output:0"=
lstm_30_while_identity_2!lstm_30/while/Identity_2:output:0"=
lstm_30_while_identity_3!lstm_30/while/Identity_3:output:0"=
lstm_30_while_identity_4!lstm_30/while/Identity_4:output:0"=
lstm_30_while_identity_5!lstm_30/while/Identity_5:output:0"P
%lstm_30_while_lstm_30_strided_slice_1'lstm_30_while_lstm_30_strided_slice_1_0"z
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"|
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"x
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"?
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2d
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2h
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_sequential_13_layer_call_fn_1258071

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:
??
	unknown_7:	?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_12578322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
while_body_1257479
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261051

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1256944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?%
?
while_body_1256250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_31_1256274_0:
??0
while_lstm_cell_31_1256276_0:
??+
while_lstm_cell_31_1256278_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_31_1256274:
??.
while_lstm_cell_31_1256276:
??)
while_lstm_cell_31_1256278:	???*while/lstm_cell_31/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_1256274_0while_lstm_cell_31_1256276_0while_lstm_cell_31_1256278_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12561802,
*while/lstm_cell_31/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_31_1256274while_lstm_cell_31_1256274_0":
while_lstm_cell_31_1256276while_lstm_cell_31_1256276_0":
while_lstm_cell_31_1256278while_lstm_cell_31_1256278_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_1260444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1257375

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1257291*
condR
while_cond_1257290*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1260586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260586___redundant_placeholder05
1while_while_cond_1260586___redundant_placeholder15
1while_while_cond_1260586___redundant_placeholder25
1while_while_cond_1260586___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_dense_32_layer_call_fn_1261060

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_12571172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257832

inputs"
lstm_29_1257796:	?#
lstm_29_1257798:
??
lstm_29_1257800:	?#
lstm_30_1257804:
??#
lstm_30_1257806:
??
lstm_30_1257808:	?#
lstm_31_1257812:
??#
lstm_31_1257814:
??
lstm_31_1257816:	?$
dense_31_1257820:
??
dense_31_1257822:	?#
dense_32_1257826:	?
dense_32_1257828:
identity?? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?"dropout_49/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?lstm_29/StatefulPartitionedCall?lstm_30/StatefulPartitionedCall?lstm_31/StatefulPartitionedCall?
lstm_29/StatefulPartitionedCallStatefulPartitionedCallinputslstm_29_1257796lstm_29_1257798lstm_29_1257800*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12577512!
lstm_29/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12575922$
"dropout_47/StatefulPartitionedCall?
lstm_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0lstm_30_1257804lstm_30_1257806lstm_30_1257808*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12575632!
lstm_30/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_30/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12574042$
"dropout_48/StatefulPartitionedCall?
lstm_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0lstm_31_1257812lstm_31_1257814lstm_31_1257816*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12573752!
lstm_31/StatefulPartitionedCall?
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall(lstm_31/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12572162$
"dropout_49/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_31_1257820dense_31_1257822*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_12570742"
 dense_31/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12571832$
"dropout_50/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_32_1257826dense_32_1257828*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_12571172"
 dense_32/StatefulPartitionedCall?
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1255721

inputs(
lstm_cell_30_1255639:
??(
lstm_cell_30_1255641:
??#
lstm_cell_30_1255643:	?
identity??$lstm_cell_30/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_1255639lstm_cell_30_1255641lstm_cell_30_1255643*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12555822&
$lstm_cell_30/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_1255639lstm_cell_30_1255641lstm_cell_30_1255643*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1255652*
condR
while_cond_1255651*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?1
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257970
lstm_29_input"
lstm_29_1257934:	?#
lstm_29_1257936:
??
lstm_29_1257938:	?#
lstm_30_1257942:
??#
lstm_30_1257944:
??
lstm_30_1257946:	?#
lstm_31_1257950:
??#
lstm_31_1257952:
??
lstm_31_1257954:	?$
dense_31_1257958:
??
dense_31_1257960:	?#
dense_32_1257964:	?
dense_32_1257966:
identity?? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?"dropout_49/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?lstm_29/StatefulPartitionedCall?lstm_30/StatefulPartitionedCall?lstm_31/StatefulPartitionedCall?
lstm_29/StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputlstm_29_1257934lstm_29_1257936lstm_29_1257938*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12577512!
lstm_29/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12575922$
"dropout_47/StatefulPartitionedCall?
lstm_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0lstm_30_1257942lstm_30_1257944lstm_30_1257946*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12575632!
lstm_30/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_30/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12574042$
"dropout_48/StatefulPartitionedCall?
lstm_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0lstm_31_1257950lstm_31_1257952lstm_31_1257954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12573752!
lstm_31/StatefulPartitionedCall?
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall(lstm_31/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12572162$
"dropout_49/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_31_1257958dense_31_1257960*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_12570742"
 dense_31/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12571832$
"dropout_50/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_32_1257964dense_32_1257966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_12571172"
 dense_32/StatefulPartitionedCall?
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?
?
/__inference_sequential_13_layer_call_fn_1257153
lstm_29_input
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:
??
	unknown_7:	?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_12571242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?%
?
while_body_1255054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_1255078_0:	?0
while_lstm_cell_29_1255080_0:
??+
while_lstm_cell_29_1255082_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_1255078:	?.
while_lstm_cell_29_1255080:
??)
while_lstm_cell_29_1255082:	???*while/lstm_cell_29/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_1255078_0while_lstm_cell_29_1255080_0while_lstm_cell_29_1255082_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12549842,
*while/lstm_cell_29/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_29_1255078while_lstm_cell_29_1255078_0":
while_lstm_cell_29_1255080while_lstm_cell_29_1255080_0":
while_lstm_cell_29_1255082while_lstm_cell_29_1255082_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
(sequential_13_lstm_30_while_cond_1254491H
Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counterN
Jsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations+
'sequential_13_lstm_30_while_placeholder-
)sequential_13_lstm_30_while_placeholder_1-
)sequential_13_lstm_30_while_placeholder_2-
)sequential_13_lstm_30_while_placeholder_3J
Fsequential_13_lstm_30_while_less_sequential_13_lstm_30_strided_slice_1a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1254491___redundant_placeholder0a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1254491___redundant_placeholder1a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1254491___redundant_placeholder2a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1254491___redundant_placeholder3(
$sequential_13_lstm_30_while_identity
?
 sequential_13/lstm_30/while/LessLess'sequential_13_lstm_30_while_placeholderFsequential_13_lstm_30_while_less_sequential_13_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_30/while/Less?
$sequential_13/lstm_30/while/IdentityIdentity$sequential_13/lstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_30/while/Identity"U
$sequential_13_lstm_30_while_identity-sequential_13/lstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261188

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
while_body_1259158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_1260230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1260729
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260729___redundant_placeholder05
1while_while_cond_1260729___redundant_placeholder15
1while_while_cond_1260729___redundant_placeholder25
1while_while_cond_1260729___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?U
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1257563

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1257479*
condR
while_cond_1257478*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1255053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1255053___redundant_placeholder05
1while_while_cond_1255053___redundant_placeholder15
1while_while_cond_1255053___redundant_placeholder25
1while_while_cond_1255053___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?U
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259671

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	?A
-lstm_cell_29_matmul_1_readvariableop_resource:
??;
,lstm_cell_29_biasadd_readvariableop_resource:	?
identity??#lstm_cell_29/BiasAdd/ReadVariableOp?"lstm_cell_29/MatMul/ReadVariableOp?$lstm_cell_29/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp?
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul?
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/MatMul_1?
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add?
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim?
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_29/split?
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid?
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_1?
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu?
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_1?
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/add_1?
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/Relu_1?
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_29/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1259587*
condR
while_cond_1259586*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_1256629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1256629___redundant_placeholder05
1while_while_cond_1256629___redundant_placeholder15
1while_while_cond_1256629___redundant_placeholder25
1while_while_cond_1256629___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260984

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1254921

inputs'
lstm_cell_29_1254839:	?(
lstm_cell_29_1254841:
??#
lstm_cell_29_1254843:	?
identity??$lstm_cell_29/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_1254839lstm_cell_29_1254841lstm_cell_29_1254843*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12548382&
$lstm_cell_29/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_1254839lstm_cell_29_1254841lstm_cell_29_1254843*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1254852*
condR
while_cond_1254851*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?U
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260171

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260087*
condR
while_cond_1260086*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259698

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1257291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_1257041

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1260873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
??G
3while_lstm_cell_31_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_31_biasadd_readvariableop_resource:	???)while/lstm_cell_31/BiasAdd/ReadVariableOp?(while/lstm_cell_31/MatMul/ReadVariableOp?*while/lstm_cell_31/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp?
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul?
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOp?
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/MatMul_1?
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add?
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOp?
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/BiasAdd?
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim?
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_31/split?
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid?
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_1?
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul?
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu?
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_1?
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/add_1?
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Sigmoid_2?
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/Relu_1?
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_31/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_1257667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1259055

inputsF
3lstm_29_lstm_cell_29_matmul_readvariableop_resource:	?I
5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
??C
4lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	?G
3lstm_30_lstm_cell_30_matmul_readvariableop_resource:
??I
5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
??C
4lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	?G
3lstm_31_lstm_cell_31_matmul_readvariableop_resource:
??I
5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
??C
4lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	?>
*dense_31_tensordot_readvariableop_resource:
??7
(dense_31_biasadd_readvariableop_resource:	?=
*dense_32_tensordot_readvariableop_resource:	?6
(dense_32_biasadd_readvariableop_resource:
identity??dense_31/BiasAdd/ReadVariableOp?!dense_31/Tensordot/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?!dense_32/Tensordot/ReadVariableOp?+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?*lstm_29/lstm_cell_29/MatMul/ReadVariableOp?,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?lstm_29/while?+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?*lstm_30/lstm_cell_30/MatMul/ReadVariableOp?,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?lstm_30/while?+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?*lstm_31/lstm_cell_31/MatMul/ReadVariableOp?,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?lstm_31/whileT
lstm_29/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_29/Shape?
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice/stack?
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_1?
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_2?
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slices
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_29/zeros/packed/1?
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros/packedo
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros/Const?
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/zerosw
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_29/zeros_1/packed/1?
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros_1/packeds
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros_1/Const?
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/zeros_1?
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose/perm?
lstm_29/transpose	Transposeinputslstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_29/transposeg
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
:2
lstm_29/Shape_1?
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_1/stack?
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_1?
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_2?
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slice_1?
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_29/TensorArrayV2/element_shape?
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_29/TensorArrayUnstack/TensorListFromTensor?
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_2/stack?
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_1?
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_2?
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_29/strided_slice_2?
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp?
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:02lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/MatMul?
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp?
lstm_29/lstm_cell_29/MatMul_1MatMullstm_29/zeros:output:04lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/MatMul_1?
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/MatMul:product:0'lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/add?
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp?
lstm_29/lstm_cell_29/BiasAddBiasAddlstm_29/lstm_cell_29/add:z:03lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/BiasAdd?
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_29/lstm_cell_29/split/split_dim?
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:0%lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_29/lstm_cell_29/split?
lstm_29/lstm_cell_29/SigmoidSigmoid#lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Sigmoid?
lstm_29/lstm_cell_29/Sigmoid_1Sigmoid#lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_29/lstm_cell_29/Sigmoid_1?
lstm_29/lstm_cell_29/mulMul"lstm_29/lstm_cell_29/Sigmoid_1:y:0lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul?
lstm_29/lstm_cell_29/ReluRelu#lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Relu?
lstm_29/lstm_cell_29/mul_1Mul lstm_29/lstm_cell_29/Sigmoid:y:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul_1?
lstm_29/lstm_cell_29/add_1AddV2lstm_29/lstm_cell_29/mul:z:0lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/add_1?
lstm_29/lstm_cell_29/Sigmoid_2Sigmoid#lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_29/lstm_cell_29/Sigmoid_2?
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/Relu_1?
lstm_29/lstm_cell_29/mul_2Mul"lstm_29/lstm_cell_29/Sigmoid_2:y:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_29/lstm_cell_29/mul_2?
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_29/TensorArrayV2_1/element_shape?
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2_1^
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/time?
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_29/while/maximum_iterationsz
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/while/loop_counter?
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_29_lstm_cell_29_matmul_readvariableop_resource5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_29_while_body_1258608*&
condR
lstm_29_while_cond_1258607*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_29/while?
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_29/TensorArrayV2Stack/TensorListStack?
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_29/strided_slice_3/stack?
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_29/strided_slice_3/stack_1?
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_3/stack_2?
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_29/strided_slice_3?
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose_1/perm?
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_29/transpose_1v
lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/runtimey
dropout_47/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_47/dropout/Const?
dropout_47/dropout/MulMullstm_29/transpose_1:y:0!dropout_47/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_47/dropout/Mul{
dropout_47/dropout/ShapeShapelstm_29/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_47/dropout/Shape?
/dropout_47/dropout/random_uniform/RandomUniformRandomUniform!dropout_47/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_47/dropout/random_uniform/RandomUniform?
!dropout_47/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_47/dropout/GreaterEqual/y?
dropout_47/dropout/GreaterEqualGreaterEqual8dropout_47/dropout/random_uniform/RandomUniform:output:0*dropout_47/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_47/dropout/GreaterEqual?
dropout_47/dropout/CastCast#dropout_47/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_47/dropout/Cast?
dropout_47/dropout/Mul_1Muldropout_47/dropout/Mul:z:0dropout_47/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_47/dropout/Mul_1j
lstm_30/ShapeShapedropout_47/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_30/Shape?
lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice/stack?
lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_1?
lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_2?
lstm_30/strided_sliceStridedSlicelstm_30/Shape:output:0$lstm_30/strided_slice/stack:output:0&lstm_30/strided_slice/stack_1:output:0&lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slices
lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_30/zeros/packed/1?
lstm_30/zeros/packedPacklstm_30/strided_slice:output:0lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros/packedo
lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros/Const?
lstm_30/zerosFilllstm_30/zeros/packed:output:0lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/zerosw
lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_30/zeros_1/packed/1?
lstm_30/zeros_1/packedPacklstm_30/strided_slice:output:0!lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros_1/packeds
lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros_1/Const?
lstm_30/zeros_1Filllstm_30/zeros_1/packed:output:0lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/zeros_1?
lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose/perm?
lstm_30/transpose	Transposedropout_47/dropout/Mul_1:z:0lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_30/transposeg
lstm_30/Shape_1Shapelstm_30/transpose:y:0*
T0*
_output_shapes
:2
lstm_30/Shape_1?
lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_1/stack?
lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_1?
lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_2?
lstm_30/strided_slice_1StridedSlicelstm_30/Shape_1:output:0&lstm_30/strided_slice_1/stack:output:0(lstm_30/strided_slice_1/stack_1:output:0(lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slice_1?
#lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_30/TensorArrayV2/element_shape?
lstm_30/TensorArrayV2TensorListReserve,lstm_30/TensorArrayV2/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_30/transpose:y:0Flstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_30/TensorArrayUnstack/TensorListFromTensor?
lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_2/stack?
lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_1?
lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_2?
lstm_30/strided_slice_2StridedSlicelstm_30/transpose:y:0&lstm_30/strided_slice_2/stack:output:0(lstm_30/strided_slice_2/stack_1:output:0(lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_30/strided_slice_2?
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp?
lstm_30/lstm_cell_30/MatMulMatMul lstm_30/strided_slice_2:output:02lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/MatMul?
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_30/lstm_cell_30/MatMul_1MatMullstm_30/zeros:output:04lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/MatMul_1?
lstm_30/lstm_cell_30/addAddV2%lstm_30/lstm_cell_30/MatMul:product:0'lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/add?
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_30/lstm_cell_30/BiasAddBiasAddlstm_30/lstm_cell_30/add:z:03lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/BiasAdd?
$lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_30/lstm_cell_30/split/split_dim?
lstm_30/lstm_cell_30/splitSplit-lstm_30/lstm_cell_30/split/split_dim:output:0%lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_30/lstm_cell_30/split?
lstm_30/lstm_cell_30/SigmoidSigmoid#lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Sigmoid?
lstm_30/lstm_cell_30/Sigmoid_1Sigmoid#lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_30/lstm_cell_30/Sigmoid_1?
lstm_30/lstm_cell_30/mulMul"lstm_30/lstm_cell_30/Sigmoid_1:y:0lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul?
lstm_30/lstm_cell_30/ReluRelu#lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Relu?
lstm_30/lstm_cell_30/mul_1Mul lstm_30/lstm_cell_30/Sigmoid:y:0'lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul_1?
lstm_30/lstm_cell_30/add_1AddV2lstm_30/lstm_cell_30/mul:z:0lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/add_1?
lstm_30/lstm_cell_30/Sigmoid_2Sigmoid#lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_30/lstm_cell_30/Sigmoid_2?
lstm_30/lstm_cell_30/Relu_1Relulstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/Relu_1?
lstm_30/lstm_cell_30/mul_2Mul"lstm_30/lstm_cell_30/Sigmoid_2:y:0)lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_30/lstm_cell_30/mul_2?
%lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_30/TensorArrayV2_1/element_shape?
lstm_30/TensorArrayV2_1TensorListReserve.lstm_30/TensorArrayV2_1/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2_1^
lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/time?
 lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_30/while/maximum_iterationsz
lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/while/loop_counter?
lstm_30/whileWhile#lstm_30/while/loop_counter:output:0)lstm_30/while/maximum_iterations:output:0lstm_30/time:output:0 lstm_30/TensorArrayV2_1:handle:0lstm_30/zeros:output:0lstm_30/zeros_1:output:0 lstm_30/strided_slice_1:output:0?lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_30_lstm_cell_30_matmul_readvariableop_resource5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_30_while_body_1258755*&
condR
lstm_30_while_cond_1258754*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_30/while?
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_30/TensorArrayV2Stack/TensorListStackTensorListStacklstm_30/while:output:3Alstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_30/TensorArrayV2Stack/TensorListStack?
lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_30/strided_slice_3/stack?
lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_30/strided_slice_3/stack_1?
lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_3/stack_2?
lstm_30/strided_slice_3StridedSlice3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_30/strided_slice_3/stack:output:0(lstm_30/strided_slice_3/stack_1:output:0(lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_30/strided_slice_3?
lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose_1/perm?
lstm_30/transpose_1	Transpose3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_30/transpose_1v
lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/runtimey
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_48/dropout/Const?
dropout_48/dropout/MulMullstm_30/transpose_1:y:0!dropout_48/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_48/dropout/Mul{
dropout_48/dropout/ShapeShapelstm_30/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_48/dropout/Shape?
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_48/dropout/random_uniform/RandomUniform?
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_48/dropout/GreaterEqual/y?
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_48/dropout/GreaterEqual?
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_48/dropout/Cast?
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_48/dropout/Mul_1j
lstm_31/ShapeShapedropout_48/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_31/Shape?
lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice/stack?
lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_1?
lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_2?
lstm_31/strided_sliceStridedSlicelstm_31/Shape:output:0$lstm_31/strided_slice/stack:output:0&lstm_31/strided_slice/stack_1:output:0&lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slices
lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_31/zeros/packed/1?
lstm_31/zeros/packedPacklstm_31/strided_slice:output:0lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros/packedo
lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros/Const?
lstm_31/zerosFilllstm_31/zeros/packed:output:0lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/zerosw
lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_31/zeros_1/packed/1?
lstm_31/zeros_1/packedPacklstm_31/strided_slice:output:0!lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros_1/packeds
lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros_1/Const?
lstm_31/zeros_1Filllstm_31/zeros_1/packed:output:0lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/zeros_1?
lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose/perm?
lstm_31/transpose	Transposedropout_48/dropout/Mul_1:z:0lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_31/transposeg
lstm_31/Shape_1Shapelstm_31/transpose:y:0*
T0*
_output_shapes
:2
lstm_31/Shape_1?
lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_1/stack?
lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_1?
lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_2?
lstm_31/strided_slice_1StridedSlicelstm_31/Shape_1:output:0&lstm_31/strided_slice_1/stack:output:0(lstm_31/strided_slice_1/stack_1:output:0(lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slice_1?
#lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_31/TensorArrayV2/element_shape?
lstm_31/TensorArrayV2TensorListReserve,lstm_31/TensorArrayV2/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_31/transpose:y:0Flstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_31/TensorArrayUnstack/TensorListFromTensor?
lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_2/stack?
lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_1?
lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_2?
lstm_31/strided_slice_2StridedSlicelstm_31/transpose:y:0&lstm_31/strided_slice_2/stack:output:0(lstm_31/strided_slice_2/stack_1:output:0(lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_31/strided_slice_2?
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp?
lstm_31/lstm_cell_31/MatMulMatMul lstm_31/strided_slice_2:output:02lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/MatMul?
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_31/lstm_cell_31/MatMul_1MatMullstm_31/zeros:output:04lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/MatMul_1?
lstm_31/lstm_cell_31/addAddV2%lstm_31/lstm_cell_31/MatMul:product:0'lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/add?
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_31/lstm_cell_31/BiasAddBiasAddlstm_31/lstm_cell_31/add:z:03lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/BiasAdd?
$lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_31/lstm_cell_31/split/split_dim?
lstm_31/lstm_cell_31/splitSplit-lstm_31/lstm_cell_31/split/split_dim:output:0%lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_31/lstm_cell_31/split?
lstm_31/lstm_cell_31/SigmoidSigmoid#lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Sigmoid?
lstm_31/lstm_cell_31/Sigmoid_1Sigmoid#lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_31/lstm_cell_31/Sigmoid_1?
lstm_31/lstm_cell_31/mulMul"lstm_31/lstm_cell_31/Sigmoid_1:y:0lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul?
lstm_31/lstm_cell_31/ReluRelu#lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Relu?
lstm_31/lstm_cell_31/mul_1Mul lstm_31/lstm_cell_31/Sigmoid:y:0'lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul_1?
lstm_31/lstm_cell_31/add_1AddV2lstm_31/lstm_cell_31/mul:z:0lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/add_1?
lstm_31/lstm_cell_31/Sigmoid_2Sigmoid#lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_31/lstm_cell_31/Sigmoid_2?
lstm_31/lstm_cell_31/Relu_1Relulstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/Relu_1?
lstm_31/lstm_cell_31/mul_2Mul"lstm_31/lstm_cell_31/Sigmoid_2:y:0)lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_31/lstm_cell_31/mul_2?
%lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_31/TensorArrayV2_1/element_shape?
lstm_31/TensorArrayV2_1TensorListReserve.lstm_31/TensorArrayV2_1/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2_1^
lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/time?
 lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_31/while/maximum_iterationsz
lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/while/loop_counter?
lstm_31/whileWhile#lstm_31/while/loop_counter:output:0)lstm_31/while/maximum_iterations:output:0lstm_31/time:output:0 lstm_31/TensorArrayV2_1:handle:0lstm_31/zeros:output:0lstm_31/zeros_1:output:0 lstm_31/strided_slice_1:output:0?lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_31_lstm_cell_31_matmul_readvariableop_resource5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_31_while_body_1258902*&
condR
lstm_31_while_cond_1258901*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_31/while?
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_31/TensorArrayV2Stack/TensorListStackTensorListStacklstm_31/while:output:3Alstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_31/TensorArrayV2Stack/TensorListStack?
lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_31/strided_slice_3/stack?
lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_31/strided_slice_3/stack_1?
lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_3/stack_2?
lstm_31/strided_slice_3StridedSlice3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_31/strided_slice_3/stack:output:0(lstm_31/strided_slice_3/stack_1:output:0(lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_31/strided_slice_3?
lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose_1/perm?
lstm_31/transpose_1	Transpose3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_31/transpose_1v
lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/runtimey
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_49/dropout/Const?
dropout_49/dropout/MulMullstm_31/transpose_1:y:0!dropout_49/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_49/dropout/Mul{
dropout_49/dropout/ShapeShapelstm_31/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_49/dropout/Shape?
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_49/dropout/random_uniform/RandomUniform?
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_49/dropout/GreaterEqual/y?
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_49/dropout/GreaterEqual?
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_49/dropout/Cast?
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_49/dropout/Mul_1?
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes?
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free?
dense_31/Tensordot/ShapeShapedropout_49/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape?
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axis?
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2?
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis?
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const?
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod?
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1?
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1?
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axis?
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat?
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stack?
dense_31/Tensordot/transpose	Transposedropout_49/dropout/Mul_1:z:0"dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Tensordot/transpose?
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_31/Tensordot/Reshape?
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/Tensordot/MatMul?
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_31/Tensordot/Const_2?
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axis?
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1?
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Tensordot?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_31/BiasAddx
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_31/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_50/dropout/Const?
dropout_50/dropout/MulMuldense_31/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_50/dropout/Mul
dropout_50/dropout/ShapeShapedense_31/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/Shape?
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_50/dropout/random_uniform/RandomUniform?
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_50/dropout/GreaterEqual/y?
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_50/dropout/GreaterEqual?
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_50/dropout/Cast?
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_50/dropout/Mul_1?
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes?
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/free?
dense_32/Tensordot/ShapeShapedropout_50/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape?
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axis?
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2?
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis?
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const?
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod?
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1?
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1?
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axis?
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat?
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stack?
dense_32/Tensordot/transpose	Transposedropout_50/dropout/Mul_1:z:0"dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_32/Tensordot/transpose?
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_32/Tensordot/Reshape?
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/Tensordot/MatMul?
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/Const_2?
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axis?
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1?
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_32/Tensordot?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_32/BiasAddx
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp"^dense_32/Tensordot/ReadVariableOp,^lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+^lstm_29/lstm_cell_29/MatMul/ReadVariableOp-^lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^lstm_29/while,^lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+^lstm_30/lstm_cell_30/MatMul/ReadVariableOp-^lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^lstm_30/while,^lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+^lstm_31/lstm_cell_31/MatMul/ReadVariableOp-^lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2F
!dense_32/Tensordot/ReadVariableOp!dense_32/Tensordot/ReadVariableOp2Z
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2X
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp*lstm_29/lstm_cell_29/MatMul/ReadVariableOp2\
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2
lstm_29/whilelstm_29/while2Z
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2X
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp*lstm_30/lstm_cell_30/MatMul/ReadVariableOp2\
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2
lstm_30/whilelstm_30/while2Z
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2X
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp*lstm_31/lstm_cell_31/MatMul/ReadVariableOp2\
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2
lstm_31/whilelstm_31/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_lstm_29_layer_call_fn_1259077
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12551232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?V
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260528
inputs_0?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260444*
condR
while_cond_1260443*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1256319

inputs(
lstm_cell_31_1256237:
??(
lstm_cell_31_1256239:
??#
lstm_cell_31_1256241:	?
identity??$lstm_cell_31/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_1256237lstm_cell_31_1256239lstm_cell_31_1256241*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12561802&
$lstm_cell_31/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_1256237lstm_cell_31_1256239lstm_cell_31_1256241*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1256250*
condR
while_cond_1256249*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260972

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_50_layer_call_fn_1261034

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12571832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1255123

inputs'
lstm_cell_29_1255041:	?(
lstm_cell_29_1255043:
??#
lstm_cell_29_1255045:	?
identity??$lstm_cell_29/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_1255041lstm_cell_29_1255043lstm_cell_29_1255045*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12549842&
$lstm_cell_29/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_1255041lstm_cell_29_1255043lstm_cell_29_1255045*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1255054*
condR
while_cond_1255053*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
)__inference_lstm_29_layer_call_fn_1259088

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12567142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260329

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1259157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259157___redundant_placeholder05
1while_while_cond_1259157___redundant_placeholder15
1while_while_cond_1259157___redundant_placeholder25
1while_while_cond_1259157___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_31_layer_call_fn_1260363
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12563192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1256117

inputs(
lstm_cell_31_1256035:
??(
lstm_cell_31_1256037:
??#
lstm_cell_31_1256039:	?
identity??$lstm_cell_31/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_1256035lstm_cell_31_1256037lstm_cell_31_1256039*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_12560342&
$lstm_cell_31/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_1256035lstm_cell_31_1256037lstm_cell_31_1256039*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1256048*
condR
while_cond_1256047*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?%
?
while_body_1255450
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_30_1255474_0:
??0
while_lstm_cell_30_1255476_0:
??+
while_lstm_cell_30_1255478_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_30_1255474:
??.
while_lstm_cell_30_1255476:
??)
while_lstm_cell_30_1255478:	???*while/lstm_cell_30/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_1255474_0while_lstm_cell_30_1255476_0while_lstm_cell_30_1255478_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12554362,
*while/lstm_cell_30/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_30_1255474while_lstm_cell_30_1255474_0":
while_lstm_cell_30_1255476while_lstm_cell_30_1255476_0":
while_lstm_cell_30_1255478while_lstm_cell_30_1255478_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_29_while_cond_1258607,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1E
Alstm_29_while_lstm_29_while_cond_1258607___redundant_placeholder0E
Alstm_29_while_lstm_29_while_cond_1258607___redundant_placeholder1E
Alstm_29_while_lstm_29_while_cond_1258607___redundant_placeholder2E
Alstm_29_while_lstm_29_while_cond_1258607___redundant_placeholder3
lstm_29_while_identity
?
lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2
lstm_29/while/Lessu
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_29/while/Identity"9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_1256787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_31_while_cond_1258409,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3.
*lstm_31_while_less_lstm_31_strided_slice_1E
Alstm_31_while_lstm_31_while_cond_1258409___redundant_placeholder0E
Alstm_31_while_lstm_31_while_cond_1258409___redundant_placeholder1E
Alstm_31_while_lstm_31_while_cond_1258409___redundant_placeholder2E
Alstm_31_while_lstm_31_while_cond_1258409___redundant_placeholder3
lstm_31_while_identity
?
lstm_31/while/LessLesslstm_31_while_placeholder*lstm_31_while_less_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2
lstm_31/while/Lessu
lstm_31/while/IdentityIdentitylstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_31/while/Identity"9
lstm_31_while_identitylstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_29_layer_call_fn_1259066
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12549212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_1257216

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1256180

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
? 
#__inference__traced_restore_1261705
file_prefix4
 assignvariableop_dense_31_kernel:
??/
 assignvariableop_1_dense_31_bias:	?5
"assignvariableop_2_dense_32_kernel:	?.
 assignvariableop_3_dense_32_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_29_lstm_cell_29_kernel:	?M
9assignvariableop_10_lstm_29_lstm_cell_29_recurrent_kernel:
??<
-assignvariableop_11_lstm_29_lstm_cell_29_bias:	?C
/assignvariableop_12_lstm_30_lstm_cell_30_kernel:
??M
9assignvariableop_13_lstm_30_lstm_cell_30_recurrent_kernel:
??<
-assignvariableop_14_lstm_30_lstm_cell_30_bias:	?C
/assignvariableop_15_lstm_31_lstm_cell_31_kernel:
??M
9assignvariableop_16_lstm_31_lstm_cell_31_recurrent_kernel:
??<
-assignvariableop_17_lstm_31_lstm_cell_31_bias:	?#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: >
*assignvariableop_22_adam_dense_31_kernel_m:
??7
(assignvariableop_23_adam_dense_31_bias_m:	?=
*assignvariableop_24_adam_dense_32_kernel_m:	?6
(assignvariableop_25_adam_dense_32_bias_m:I
6assignvariableop_26_adam_lstm_29_lstm_cell_29_kernel_m:	?T
@assignvariableop_27_adam_lstm_29_lstm_cell_29_recurrent_kernel_m:
??C
4assignvariableop_28_adam_lstm_29_lstm_cell_29_bias_m:	?J
6assignvariableop_29_adam_lstm_30_lstm_cell_30_kernel_m:
??T
@assignvariableop_30_adam_lstm_30_lstm_cell_30_recurrent_kernel_m:
??C
4assignvariableop_31_adam_lstm_30_lstm_cell_30_bias_m:	?J
6assignvariableop_32_adam_lstm_31_lstm_cell_31_kernel_m:
??T
@assignvariableop_33_adam_lstm_31_lstm_cell_31_recurrent_kernel_m:
??C
4assignvariableop_34_adam_lstm_31_lstm_cell_31_bias_m:	?>
*assignvariableop_35_adam_dense_31_kernel_v:
??7
(assignvariableop_36_adam_dense_31_bias_v:	?=
*assignvariableop_37_adam_dense_32_kernel_v:	?6
(assignvariableop_38_adam_dense_32_bias_v:I
6assignvariableop_39_adam_lstm_29_lstm_cell_29_kernel_v:	?T
@assignvariableop_40_adam_lstm_29_lstm_cell_29_recurrent_kernel_v:
??C
4assignvariableop_41_adam_lstm_29_lstm_cell_29_bias_v:	?J
6assignvariableop_42_adam_lstm_30_lstm_cell_30_kernel_v:
??T
@assignvariableop_43_adam_lstm_30_lstm_cell_30_recurrent_kernel_v:
??C
4assignvariableop_44_adam_lstm_30_lstm_cell_30_bias_v:	?J
6assignvariableop_45_adam_lstm_31_lstm_cell_31_kernel_v:
??T
@assignvariableop_46_adam_lstm_31_lstm_cell_31_recurrent_kernel_v:
??C
4assignvariableop_47_adam_lstm_31_lstm_cell_31_bias_v:	?
identity_49??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_32_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_32_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_29_lstm_cell_29_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_29_lstm_cell_29_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_29_lstm_cell_29_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_lstm_30_lstm_cell_30_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_lstm_30_lstm_cell_30_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_30_lstm_cell_30_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_lstm_31_lstm_cell_31_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_lstm_31_lstm_cell_31_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_lstm_31_lstm_cell_31_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_31_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_31_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_32_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_32_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_lstm_29_lstm_cell_29_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_lstm_29_lstm_cell_29_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_29_lstm_cell_29_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_30_lstm_cell_30_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_30_lstm_cell_30_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_30_lstm_cell_30_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_31_lstm_cell_31_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_31_lstm_cell_31_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_31_lstm_cell_31_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_31_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_31_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_32_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_32_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_lstm_29_lstm_cell_29_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_lstm_29_lstm_cell_29_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_lstm_29_lstm_cell_29_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_lstm_30_lstm_cell_30_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_lstm_30_lstm_cell_30_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_lstm_30_lstm_cell_30_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_lstm_31_lstm_cell_31_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_lstm_31_lstm_cell_31_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_lstm_31_lstm_cell_31_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_479
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48f
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_49?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
/__inference_sequential_13_layer_call_fn_1257892
lstm_29_input
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:
??
	unknown_7:	?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_12578322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_29_input
?
?
while_cond_1259443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1259443___redundant_placeholder05
1while_while_cond_1259443___redundant_placeholder15
1while_while_cond_1259443___redundant_placeholder25
1while_while_cond_1259443___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?%
?
while_body_1255652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_30_1255676_0:
??0
while_lstm_cell_30_1255678_0:
??+
while_lstm_cell_30_1255680_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_30_1255676:
??.
while_lstm_cell_30_1255678:
??)
while_lstm_cell_30_1255680:	???*while/lstm_cell_30/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_1255676_0while_lstm_cell_30_1255678_0while_lstm_cell_30_1255680_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12555822,
*while/lstm_cell_30/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_30_1255676while_lstm_cell_30_1255676_0":
while_lstm_cell_30_1255678while_lstm_cell_30_1255678_0":
while_lstm_cell_30_1255680while_lstm_cell_30_1255680_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
(sequential_13_lstm_29_while_cond_1254351H
Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counterN
Jsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations+
'sequential_13_lstm_29_while_placeholder-
)sequential_13_lstm_29_while_placeholder_1-
)sequential_13_lstm_29_while_placeholder_2-
)sequential_13_lstm_29_while_placeholder_3J
Fsequential_13_lstm_29_while_less_sequential_13_lstm_29_strided_slice_1a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1254351___redundant_placeholder0a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1254351___redundant_placeholder1a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1254351___redundant_placeholder2a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1254351___redundant_placeholder3(
$sequential_13_lstm_29_while_identity
?
 sequential_13/lstm_29/while/LessLess'sequential_13_lstm_29_while_placeholderFsequential_13_lstm_29_while_less_sequential_13_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_29/while/Less?
$sequential_13/lstm_29/while/IdentityIdentity$sequential_13/lstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_29/while/Identity"U
$sequential_13_lstm_29_while_identity-sequential_13/lstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?*
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257124

inputs"
lstm_29_1256715:	?#
lstm_29_1256717:
??
lstm_29_1256719:	?#
lstm_30_1256872:
??#
lstm_30_1256874:
??
lstm_30_1256876:	?#
lstm_31_1257029:
??#
lstm_31_1257031:
??
lstm_31_1257033:	?$
dense_31_1257075:
??
dense_31_1257077:	?#
dense_32_1257118:	?
dense_32_1257120:
identity?? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall?lstm_29/StatefulPartitionedCall?lstm_30/StatefulPartitionedCall?lstm_31/StatefulPartitionedCall?
lstm_29/StatefulPartitionedCallStatefulPartitionedCallinputslstm_29_1256715lstm_29_1256717lstm_29_1256719*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_12567142!
lstm_29/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_12567272
dropout_47/PartitionedCall?
lstm_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0lstm_30_1256872lstm_30_1256874lstm_30_1256876*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_12568712!
lstm_30/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall(lstm_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_12568842
dropout_48/PartitionedCall?
lstm_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0lstm_31_1257029lstm_31_1257031lstm_31_1257033*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_12570282!
lstm_31/StatefulPartitionedCall?
dropout_49/PartitionedCallPartitionedCall(lstm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_12570412
dropout_49/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_31_1257075dense_31_1257077*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_12570742"
 dense_31/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_12570852
dropout_50/PartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_32_1257118dense_32_1257120*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_12571172"
 dense_32/StatefulPartitionedCall?
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
while_body_1256630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	?I
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	?G
3while_lstm_cell_29_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_29_biasadd_readvariableop_resource:	???)while/lstm_cell_29/BiasAdd/ReadVariableOp?(while/lstm_cell_29/MatMul/ReadVariableOp?*while/lstm_cell_29/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp?
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul?
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOp?
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/MatMul_1?
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add?
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOp?
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/BiasAdd?
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim?
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_29/split?
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid?
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_1?
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul?
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu?
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_1?
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/add_1?
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Sigmoid_2?
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/Relu_1?
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_29/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1257478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1257478___redundant_placeholder05
1while_while_cond_1257478___redundant_placeholder15
1while_while_cond_1257478___redundant_placeholder25
1while_while_cond_1257478___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1256034

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?U
?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1257028

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
??A
-lstm_cell_31_matmul_1_readvariableop_resource:
??;
,lstm_cell_31_biasadd_readvariableop_resource:	?
identity??#lstm_cell_31/BiasAdd/ReadVariableOp?"lstm_cell_31/MatMul/ReadVariableOp?$lstm_cell_31/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp?
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul?
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp?
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/MatMul_1?
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add?
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp?
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim?
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_31/split?
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid?
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_1?
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu?
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_1?
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/add_1?
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/Relu_1?
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_31/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1256944*
condR
while_cond_1256943*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_1259801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
??G
3while_lstm_cell_30_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_30_biasadd_readvariableop_resource:	???)while/lstm_cell_30/BiasAdd/ReadVariableOp?(while/lstm_cell_30/MatMul/ReadVariableOp?*while/lstm_cell_30/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp?
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul?
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOp?
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/MatMul_1?
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add?
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOp?
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/BiasAdd?
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim?
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_30/split?
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid?
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_1?
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul?
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu?
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_1?
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/add_1?
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Sigmoid_2?
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/Relu_1?
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_30/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_lstm_cell_30_layer_call_fn_1261205

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_12554362
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
*__inference_dense_31_layer_call_fn_1260993

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_12570742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261156

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_1260443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1260443___redundant_placeholder05
1while_while_cond_1260443___redundant_placeholder15
1while_while_cond_1260443___redundant_placeholder25
1while_while_cond_1260443___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
lstm_29_while_cond_1258129,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1E
Alstm_29_while_lstm_29_while_cond_1258129___redundant_placeholder0E
Alstm_29_while_lstm_29_while_cond_1258129___redundant_placeholder1E
Alstm_29_while_lstm_29_while_cond_1258129___redundant_placeholder2E
Alstm_29_while_lstm_29_while_cond_1258129___redundant_placeholder3
lstm_29_while_identity
?
lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2
lstm_29/while/Lessu
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_29/while/Identity"9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?U
?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260314

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
??A
-lstm_cell_30_matmul_1_readvariableop_resource:
??;
,lstm_cell_30_biasadd_readvariableop_resource:	?
identity??#lstm_cell_30/BiasAdd/ReadVariableOp?"lstm_cell_30/MatMul/ReadVariableOp?$lstm_cell_30/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp?
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul?
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp?
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/MatMul_1?
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add?
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp?
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim?
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_30/split?
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid?
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_1?
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu?
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_1?
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/add_1?
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/Relu_1?
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_30/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1260230*
condR
while_cond_1260229*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261286

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?!
?
E__inference_dense_31_layer_call_and_return_conditional_losses_1257074

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_lstm_cell_29_layer_call_fn_1261107

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_12548382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?^
?
(sequential_13_lstm_31_while_body_1254632H
Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counterN
Jsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations+
'sequential_13_lstm_31_while_placeholder-
)sequential_13_lstm_31_while_placeholder_1-
)sequential_13_lstm_31_while_placeholder_2-
)sequential_13_lstm_31_while_placeholder_3G
Csequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1_0?
sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
??_
Ksequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
??Y
Jsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	?(
$sequential_13_lstm_31_while_identity*
&sequential_13_lstm_31_while_identity_1*
&sequential_13_lstm_31_while_identity_2*
&sequential_13_lstm_31_while_identity_3*
&sequential_13_lstm_31_while_identity_4*
&sequential_13_lstm_31_while_identity_5E
Asequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1?
}sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
??]
Isequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
??W
Hsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	????sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
Msequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2O
Msequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_31_while_placeholderVsequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02A
?sequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem?
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp?
/sequential_13/lstm_31/while/lstm_cell_31/MatMulMatMulFsequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_31/while/lstm_cell_31/MatMul?
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02B
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp?
1sequential_13/lstm_31/while/lstm_cell_31/MatMul_1MatMul)sequential_13_lstm_31_while_placeholder_2Hsequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_13/lstm_31/while/lstm_cell_31/MatMul_1?
,sequential_13/lstm_31/while/lstm_cell_31/addAddV29sequential_13/lstm_31/while/lstm_cell_31/MatMul:product:0;sequential_13/lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_31/while/lstm_cell_31/add?
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02A
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?
0sequential_13/lstm_31/while/lstm_cell_31/BiasAddBiasAdd0sequential_13/lstm_31/while/lstm_cell_31/add:z:0Gsequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_31/while/lstm_cell_31/BiasAdd?
8sequential_13/lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_31/while/lstm_cell_31/split/split_dim?
.sequential_13/lstm_31/while/lstm_cell_31/splitSplitAsequential_13/lstm_31/while/lstm_cell_31/split/split_dim:output:09sequential_13/lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split20
.sequential_13/lstm_31/while/lstm_cell_31/split?
0sequential_13/lstm_31/while/lstm_cell_31/SigmoidSigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:??????????22
0sequential_13/lstm_31/while/lstm_cell_31/Sigmoid?
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1?
,sequential_13/lstm_31/while/lstm_cell_31/mulMul6sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1:y:0)sequential_13_lstm_31_while_placeholder_3*
T0*(
_output_shapes
:??????????2.
,sequential_13/lstm_31/while/lstm_cell_31/mul?
-sequential_13/lstm_31/while/lstm_cell_31/ReluRelu7sequential_13/lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:??????????2/
-sequential_13/lstm_31/while/lstm_cell_31/Relu?
.sequential_13/lstm_31/while/lstm_cell_31/mul_1Mul4sequential_13/lstm_31/while/lstm_cell_31/Sigmoid:y:0;sequential_13/lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_31/while/lstm_cell_31/mul_1?
.sequential_13/lstm_31/while/lstm_cell_31/add_1AddV20sequential_13/lstm_31/while/lstm_cell_31/mul:z:02sequential_13/lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_31/while/lstm_cell_31/add_1?
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:??????????24
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2?
/sequential_13/lstm_31/while/lstm_cell_31/Relu_1Relu2sequential_13/lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:??????????21
/sequential_13/lstm_31/while/lstm_cell_31/Relu_1?
.sequential_13/lstm_31/while/lstm_cell_31/mul_2Mul6sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2:y:0=sequential_13/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:??????????20
.sequential_13/lstm_31/while/lstm_cell_31/mul_2?
@sequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_31_while_placeholder_1'sequential_13_lstm_31_while_placeholder2sequential_13/lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItem?
!sequential_13/lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_31/while/add/y?
sequential_13/lstm_31/while/addAddV2'sequential_13_lstm_31_while_placeholder*sequential_13/lstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_31/while/add?
#sequential_13/lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_31/while/add_1/y?
!sequential_13/lstm_31/while/add_1AddV2Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counter,sequential_13/lstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_31/while/add_1?
$sequential_13/lstm_31/while/IdentityIdentity%sequential_13/lstm_31/while/add_1:z:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_31/while/Identity?
&sequential_13/lstm_31/while/Identity_1IdentityJsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_1?
&sequential_13/lstm_31/while/Identity_2Identity#sequential_13/lstm_31/while/add:z:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_2?
&sequential_13/lstm_31/while/Identity_3IdentityPsequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_3?
&sequential_13/lstm_31/while/Identity_4Identity2sequential_13/lstm_31/while/lstm_cell_31/mul_2:z:0!^sequential_13/lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_31/while/Identity_4?
&sequential_13/lstm_31/while/Identity_5Identity2sequential_13/lstm_31/while/lstm_cell_31/add_1:z:0!^sequential_13/lstm_31/while/NoOp*
T0*(
_output_shapes
:??????????2(
&sequential_13/lstm_31/while/Identity_5?
 sequential_13/lstm_31/while/NoOpNoOp@^sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?^sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpA^sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_31/while/NoOp"U
$sequential_13_lstm_31_while_identity-sequential_13/lstm_31/while/Identity:output:0"Y
&sequential_13_lstm_31_while_identity_1/sequential_13/lstm_31/while/Identity_1:output:0"Y
&sequential_13_lstm_31_while_identity_2/sequential_13/lstm_31/while/Identity_2:output:0"Y
&sequential_13_lstm_31_while_identity_3/sequential_13/lstm_31/while/Identity_3:output:0"Y
&sequential_13_lstm_31_while_identity_4/sequential_13/lstm_31/while/Identity_4:output:0"Y
&sequential_13_lstm_31_while_identity_5/sequential_13/lstm_31/while/Identity_5:output:0"?
Hsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resourceJsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"?
Isequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resourceKsequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"?
Gsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resourceIsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"?
Asequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1Csequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1_0"?
}sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2?
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2?
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259686

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(sequential_13_lstm_31_while_cond_1254631H
Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counterN
Jsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations+
'sequential_13_lstm_31_while_placeholder-
)sequential_13_lstm_31_while_placeholder_1-
)sequential_13_lstm_31_while_placeholder_2-
)sequential_13_lstm_31_while_placeholder_3J
Fsequential_13_lstm_31_while_less_sequential_13_lstm_31_strided_slice_1a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1254631___redundant_placeholder0a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1254631___redundant_placeholder1a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1254631___redundant_placeholder2a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1254631___redundant_placeholder3(
$sequential_13_lstm_31_while_identity
?
 sequential_13/lstm_31/while/LessLess'sequential_13_lstm_31_while_placeholderFsequential_13_lstm_31_while_less_sequential_13_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_31/while/Less?
$sequential_13/lstm_31/while/IdentityIdentity$sequential_13/lstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_31/while/Identity"U
$sequential_13_lstm_31_while_identity-sequential_13/lstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_29_input:
serving_default_lstm_29_input:0?????????@
dense_324
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$cell
%
state_spec
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
*trainable_variables
+	variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.m?/m?8m?9m?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?.v?/v?8v?9v?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?"
	optimizer
~
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912"
trackable_list_wrapper
~
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Llayers
trainable_variables
Mmetrics
	variables
regularization_losses
Nlayer_metrics
Olayer_regularization_losses
Pnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Vlayers
trainable_variables
	variables
Wmetrics
regularization_losses
Xlayer_metrics
Ylayer_regularization_losses

Zstates
[non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

\layers
trainable_variables
]metrics
	variables
regularization_losses
^layer_metrics
_layer_regularization_losses
`non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
a
state_size

Fkernel
Grecurrent_kernel
Hbias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

flayers
trainable_variables
	variables
gmetrics
regularization_losses
hlayer_metrics
ilayer_regularization_losses

jstates
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

llayers
 trainable_variables
mmetrics
!	variables
"regularization_losses
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

vlayers
&trainable_variables
'	variables
wmetrics
(regularization_losses
xlayer_metrics
ylayer_regularization_losses

zstates
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

|layers
*trainable_variables
}metrics
+	variables
,regularization_losses
~layer_metrics
layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_31/kernel
:?2dense_31/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
0trainable_variables
?metrics
1	variables
2regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
4trainable_variables
?metrics
5	variables
6regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_32/kernel
:2dense_32/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
:trainable_variables
?metrics
;	variables
<regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	?2lstm_29/lstm_cell_29/kernel
9:7
??2%lstm_29/lstm_cell_29/recurrent_kernel
(:&?2lstm_29/lstm_cell_29/bias
/:-
??2lstm_30/lstm_cell_30/kernel
9:7
??2%lstm_30/lstm_cell_30/recurrent_kernel
(:&?2lstm_30/lstm_cell_30/bias
/:-
??2lstm_31/lstm_cell_31/kernel
9:7
??2%lstm_31/lstm_cell_31/recurrent_kernel
(:&?2lstm_31/lstm_cell_31/bias
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Rtrainable_variables
?metrics
S	variables
Tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
btrainable_variables
?metrics
c	variables
dregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
rtrainable_variables
?metrics
s	variables
tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_31/kernel/m
!:?2Adam/dense_31/bias/m
':%	?2Adam/dense_32/kernel/m
 :2Adam/dense_32/bias/m
3:1	?2"Adam/lstm_29/lstm_cell_29/kernel/m
>:<
??2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
-:+?2 Adam/lstm_29/lstm_cell_29/bias/m
4:2
??2"Adam/lstm_30/lstm_cell_30/kernel/m
>:<
??2,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m
-:+?2 Adam/lstm_30/lstm_cell_30/bias/m
4:2
??2"Adam/lstm_31/lstm_cell_31/kernel/m
>:<
??2,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m
-:+?2 Adam/lstm_31/lstm_cell_31/bias/m
(:&
??2Adam/dense_31/kernel/v
!:?2Adam/dense_31/bias/v
':%	?2Adam/dense_32/kernel/v
 :2Adam/dense_32/bias/v
3:1	?2"Adam/lstm_29/lstm_cell_29/kernel/v
>:<
??2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
-:+?2 Adam/lstm_29/lstm_cell_29/bias/v
4:2
??2"Adam/lstm_30/lstm_cell_30/kernel/v
>:<
??2,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v
-:+?2 Adam/lstm_30/lstm_cell_30/bias/v
4:2
??2"Adam/lstm_31/lstm_cell_31/kernel/v
>:<
??2,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v
-:+?2 Adam/lstm_31/lstm_cell_31/bias/v
?2?
/__inference_sequential_13_layer_call_fn_1257153
/__inference_sequential_13_layer_call_fn_1258040
/__inference_sequential_13_layer_call_fn_1258071
/__inference_sequential_13_layer_call_fn_1257892?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1258549
J__inference_sequential_13_layer_call_and_return_conditional_losses_1259055
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257931
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257970?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1254771lstm_29_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_lstm_29_layer_call_fn_1259066
)__inference_lstm_29_layer_call_fn_1259077
)__inference_lstm_29_layer_call_fn_1259088
)__inference_lstm_29_layer_call_fn_1259099?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259242
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259385
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259528
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259671?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_47_layer_call_fn_1259676
,__inference_dropout_47_layer_call_fn_1259681?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259686
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lstm_30_layer_call_fn_1259709
)__inference_lstm_30_layer_call_fn_1259720
)__inference_lstm_30_layer_call_fn_1259731
)__inference_lstm_30_layer_call_fn_1259742?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1259885
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260028
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260171
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260314?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_48_layer_call_fn_1260319
,__inference_dropout_48_layer_call_fn_1260324?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260329
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260341?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lstm_31_layer_call_fn_1260352
)__inference_lstm_31_layer_call_fn_1260363
)__inference_lstm_31_layer_call_fn_1260374
)__inference_lstm_31_layer_call_fn_1260385?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260528
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260671
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260814
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260957?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_49_layer_call_fn_1260962
,__inference_dropout_49_layer_call_fn_1260967?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260972
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260984?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dense_31_layer_call_fn_1260993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_31_layer_call_and_return_conditional_losses_1261024?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dropout_50_layer_call_fn_1261029
,__inference_dropout_50_layer_call_fn_1261034?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261039
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261051?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dense_32_layer_call_fn_1261060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_32_layer_call_and_return_conditional_losses_1261090?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1258009lstm_29_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_lstm_cell_29_layer_call_fn_1261107
.__inference_lstm_cell_29_layer_call_fn_1261124?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261156
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261188?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_lstm_cell_30_layer_call_fn_1261205
.__inference_lstm_cell_30_layer_call_fn_1261222?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261254
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261286?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_lstm_cell_31_layer_call_fn_1261303
.__inference_lstm_cell_31_layer_call_fn_1261320?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261352
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261384?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
"__inference__wrapped_model_1254771?CDEFGHIJK./89:?7
0?-
+?(
lstm_29_input?????????
? "7?4
2
dense_32&?#
dense_32??????????
E__inference_dense_31_layer_call_and_return_conditional_losses_1261024f./4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
*__inference_dense_31_layer_call_fn_1260993Y./4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_dense_32_layer_call_and_return_conditional_losses_1261090e894?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
*__inference_dense_32_layer_call_fn_1261060X894?1
*?'
%?"
inputs??????????
? "???????????
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259686f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_47_layer_call_and_return_conditional_losses_1259698f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_47_layer_call_fn_1259676Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_47_layer_call_fn_1259681Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260329f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_48_layer_call_and_return_conditional_losses_1260341f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_48_layer_call_fn_1260319Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_48_layer_call_fn_1260324Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260972f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_49_layer_call_and_return_conditional_losses_1260984f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_49_layer_call_fn_1260962Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_49_layer_call_fn_1260967Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261039f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_50_layer_call_and_return_conditional_losses_1261051f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_50_layer_call_fn_1261029Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_50_layer_call_fn_1261034Y8?5
.?+
%?"
inputs??????????
p
? "????????????
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259242?CDEO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259385?CDEO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259528rCDE??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
D__inference_lstm_29_layer_call_and_return_conditional_losses_1259671rCDE??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
)__inference_lstm_29_layer_call_fn_1259066~CDEO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
)__inference_lstm_29_layer_call_fn_1259077~CDEO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
)__inference_lstm_29_layer_call_fn_1259088eCDE??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
)__inference_lstm_29_layer_call_fn_1259099eCDE??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
D__inference_lstm_30_layer_call_and_return_conditional_losses_1259885?FGHP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260028?FGHP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260171sFGH@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
D__inference_lstm_30_layer_call_and_return_conditional_losses_1260314sFGH@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
)__inference_lstm_30_layer_call_fn_1259709FGHP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
)__inference_lstm_30_layer_call_fn_1259720FGHP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
)__inference_lstm_30_layer_call_fn_1259731fFGH@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
)__inference_lstm_30_layer_call_fn_1259742fFGH@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260528?IJKP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260671?IJKP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260814sIJK@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
D__inference_lstm_31_layer_call_and_return_conditional_losses_1260957sIJK@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
)__inference_lstm_31_layer_call_fn_1260352IJKP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
)__inference_lstm_31_layer_call_fn_1260363IJKP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
)__inference_lstm_31_layer_call_fn_1260374fIJK@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
)__inference_lstm_31_layer_call_fn_1260385fIJK@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261156?CDE??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1261188?CDE??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
.__inference_lstm_cell_29_layer_call_fn_1261107?CDE??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
.__inference_lstm_cell_29_layer_call_fn_1261124?CDE??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261254?FGH???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1261286?FGH???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
.__inference_lstm_cell_30_layer_call_fn_1261205?FGH???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
.__inference_lstm_cell_30_layer_call_fn_1261222?FGH???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261352?IJK???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1261384?IJK???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
.__inference_lstm_cell_31_layer_call_fn_1261303?IJK???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
.__inference_lstm_cell_31_layer_call_fn_1261320?IJK???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257931~CDEFGHIJK./89B??
8?5
+?(
lstm_29_input?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1257970~CDEFGHIJK./89B??
8?5
+?(
lstm_29_input?????????
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1258549wCDEFGHIJK./89;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_1259055wCDEFGHIJK./89;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
/__inference_sequential_13_layer_call_fn_1257153qCDEFGHIJK./89B??
8?5
+?(
lstm_29_input?????????
p 

 
? "???????????
/__inference_sequential_13_layer_call_fn_1257892qCDEFGHIJK./89B??
8?5
+?(
lstm_29_input?????????
p

 
? "???????????
/__inference_sequential_13_layer_call_fn_1258040jCDEFGHIJK./89;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_13_layer_call_fn_1258071jCDEFGHIJK./89;?8
1?.
$?!
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_1258009?CDEFGHIJK./89K?H
? 
A?>
<
lstm_29_input+?(
lstm_29_input?????????"7?4
2
dense_32&?#
dense_32?????????