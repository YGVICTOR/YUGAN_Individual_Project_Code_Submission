¢í:
¦
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
­
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.6.02unknown8È¯8
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
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

lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø*,
shared_namelstm_20/lstm_cell_20/kernel

/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/kernel*
_output_shapes
:	Ø*
dtype0
¨
%lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*6
shared_name'%lstm_20/lstm_cell_20/recurrent_kernel
¡
9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_20/lstm_cell_20/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0

lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø**
shared_namelstm_20/lstm_cell_20/bias

-lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/bias*
_output_shapes	
:Ø*
dtype0

lstm_21/lstm_cell_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*,
shared_namelstm_21/lstm_cell_21/kernel

/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/kernel* 
_output_shapes
:
Ø*
dtype0
¨
%lstm_21/lstm_cell_21/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*6
shared_name'%lstm_21/lstm_cell_21/recurrent_kernel
¡
9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_21/lstm_cell_21/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0

lstm_21/lstm_cell_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø**
shared_namelstm_21/lstm_cell_21/bias

-lstm_21/lstm_cell_21/bias/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/bias*
_output_shapes	
:Ø*
dtype0

lstm_22/lstm_cell_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*,
shared_namelstm_22/lstm_cell_22/kernel

/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/kernel* 
_output_shapes
:
Ø*
dtype0
¨
%lstm_22/lstm_cell_22/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*6
shared_name'%lstm_22/lstm_cell_22/recurrent_kernel
¡
9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_22/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0

lstm_22/lstm_cell_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø**
shared_namelstm_22/lstm_cell_22/bias

-lstm_22/lstm_cell_22/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/bias*
_output_shapes	
:Ø*
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

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_26/kernel/m

*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_20/lstm_cell_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/m

6Adam/lstm_20/lstm_cell_20/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/m*
_output_shapes
:	Ø*
dtype0
¶
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
¯
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_20/lstm_cell_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/m

4Adam/lstm_20/lstm_cell_20/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/m*
_output_shapes	
:Ø*
dtype0
¢
"Adam/lstm_21/lstm_cell_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/m

6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/m* 
_output_shapes
:
Ø*
dtype0
¶
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
¯
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_21/lstm_cell_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/m

4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/m*
_output_shapes	
:Ø*
dtype0
¢
"Adam/lstm_22/lstm_cell_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/m

6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/m* 
_output_shapes
:
Ø*
dtype0
¶
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
¯
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_22/lstm_cell_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/m

4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/m*
_output_shapes	
:Ø*
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_26/kernel/v

*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_20/lstm_cell_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/v

6Adam/lstm_20/lstm_cell_20/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/v*
_output_shapes
:	Ø*
dtype0
¶
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
¯
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_20/lstm_cell_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/v

4Adam/lstm_20/lstm_cell_20/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/v*
_output_shapes	
:Ø*
dtype0
¢
"Adam/lstm_21/lstm_cell_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/v

6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/v* 
_output_shapes
:
Ø*
dtype0
¶
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
¯
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_21/lstm_cell_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/v

4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/v*
_output_shapes	
:Ø*
dtype0
¢
"Adam/lstm_22/lstm_cell_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/v

6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/v* 
_output_shapes
:
Ø*
dtype0
¶
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
¯
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v* 
_output_shapes
:
Ø*
dtype0

 Adam/lstm_22/lstm_cell_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/v

4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/v*
_output_shapes	
:Ø*
dtype0

NoOpNoOp
ëU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¦U
valueUBU BU
è
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
l
$cell
%
state_spec
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
Ä
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.mª/m«8m¬9m­Cm®Dm¯Em°Fm±Gm²Hm³Im´JmµKm¶.v·/v¸8v¹9vºCv»Dv¼Ev½Fv¾Gv¿HvÀIvÁJvÂKvÃ
 
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
­

Llayers
Mlayer_regularization_losses
regularization_losses
Nmetrics
trainable_variables
	variables
Olayer_metrics
Pnon_trainable_variables
 

Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
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
¹

Vlayers
Wlayer_regularization_losses

Xstates
regularization_losses
Ymetrics
trainable_variables
	variables
Zlayer_metrics
[non_trainable_variables
 
 
 
­

\layers
]layer_regularization_losses
^metrics
regularization_losses
trainable_variables
	variables
_layer_metrics
`non_trainable_variables

a
state_size

Fkernel
Grecurrent_kernel
Hbias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
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
¹

flayers
glayer_regularization_losses

hstates
regularization_losses
imetrics
trainable_variables
	variables
jlayer_metrics
knon_trainable_variables
 
 
 
­

llayers
mlayer_regularization_losses
nmetrics
 regularization_losses
!trainable_variables
"	variables
olayer_metrics
pnon_trainable_variables

q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
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
¹

vlayers
wlayer_regularization_losses

xstates
&regularization_losses
ymetrics
'trainable_variables
(	variables
zlayer_metrics
{non_trainable_variables
 
 
 
®

|layers
}layer_regularization_losses
~metrics
*regularization_losses
+trainable_variables
,	variables
layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
²
layers
 layer_regularization_losses
metrics
0regularization_losses
1trainable_variables
2	variables
layer_metrics
non_trainable_variables
 
 
 
²
layers
 layer_regularization_losses
metrics
4regularization_losses
5trainable_variables
6	variables
layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
²
layers
 layer_regularization_losses
metrics
:regularization_losses
;trainable_variables
<	variables
layer_metrics
non_trainable_variables
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
VARIABLE_VALUElstm_20/lstm_cell_20/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_20/lstm_cell_20/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_20/lstm_cell_20/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_21/lstm_cell_21/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_21/lstm_cell_21/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_21/lstm_cell_21/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_22/lstm_cell_22/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_22/lstm_cell_22/recurrent_kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_22/lstm_cell_22/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
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
 

0
1
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
²
layers
 layer_regularization_losses
metrics
Rregularization_losses
Strainable_variables
T	variables
layer_metrics
non_trainable_variables
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
 

F0
G1
H2

F0
G1
H2
²
layers
 layer_regularization_losses
metrics
bregularization_losses
ctrainable_variables
d	variables
layer_metrics
non_trainable_variables
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
 

I0
J1
K2

I0
J1
K2
²
layers
 layer_regularization_losses
metrics
rregularization_losses
strainable_variables
t	variables
layer_metrics
 non_trainable_variables
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

¡total

¢count
£	variables
¤	keras_api
I

¥total

¦count
§
_fn_kwargs
¨	variables
©	keras_api
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
¡0
¢1

£	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

¥0
¦1

¨	variables
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_20_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_20_inputlstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biaslstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biaslstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_756003
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
½
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOp9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOp-lstm_20/lstm_cell_20/bias/Read/ReadVariableOp/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOp9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOp-lstm_21/lstm_cell_21/bias/Read/ReadVariableOp/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOp9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOp-lstm_22/lstm_cell_22/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp6Adam/lstm_20/lstm_cell_20/kernel/m/Read/ReadVariableOp@Adam/lstm_20/lstm_cell_20/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_20/lstm_cell_20/bias/m/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp6Adam/lstm_20/lstm_cell_20/kernel/v/Read/ReadVariableOp@Adam/lstm_20/lstm_cell_20/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_20/lstm_cell_20/bias/v/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOpConst*=
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
GPU 2J 8 *(
f#R!
__inference__traced_save_759545
ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_25/kerneldense_25/biasdense_26/kerneldense_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biaslstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biaslstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biastotalcounttotal_1count_1Adam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/m"Adam/lstm_20/lstm_cell_20/kernel/m,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m Adam/lstm_20/lstm_cell_20/bias/m"Adam/lstm_21/lstm_cell_21/kernel/m,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m Adam/lstm_21/lstm_cell_21/bias/m"Adam/lstm_22/lstm_cell_22/kernel/m,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m Adam/lstm_22/lstm_cell_22/bias/mAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/v"Adam/lstm_20/lstm_cell_20/kernel/v,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v Adam/lstm_20/lstm_cell_20/bias/v"Adam/lstm_21/lstm_cell_21/kernel/v,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v Adam/lstm_21/lstm_cell_21/bias/v"Adam/lstm_22/lstm_cell_22/kernel/v,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v Adam/lstm_22/lstm_cell_22/bias/v*<
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_759699Ç6
Æ
ø
-__inference_lstm_cell_22_layer_call_fn_759297

inputs
states_0
states_1
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7540282
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
æ^

'sequential_10_lstm_22_while_body_752626H
Dsequential_10_lstm_22_while_sequential_10_lstm_22_while_loop_counterN
Jsequential_10_lstm_22_while_sequential_10_lstm_22_while_maximum_iterations+
'sequential_10_lstm_22_while_placeholder-
)sequential_10_lstm_22_while_placeholder_1-
)sequential_10_lstm_22_while_placeholder_2-
)sequential_10_lstm_22_while_placeholder_3G
Csequential_10_lstm_22_while_sequential_10_lstm_22_strided_slice_1_0
sequential_10_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_22_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_10_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:
Ø_
Ksequential_10_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØY
Jsequential_10_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø(
$sequential_10_lstm_22_while_identity*
&sequential_10_lstm_22_while_identity_1*
&sequential_10_lstm_22_while_identity_2*
&sequential_10_lstm_22_while_identity_3*
&sequential_10_lstm_22_while_identity_4*
&sequential_10_lstm_22_while_identity_5E
Asequential_10_lstm_22_while_sequential_10_lstm_22_strided_slice_1
}sequential_10_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_22_tensorarrayunstack_tensorlistfromtensor[
Gsequential_10_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:
Ø]
Isequential_10_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ØW
Hsequential_10_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢?sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢>sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢@sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpï
Msequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_22_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_22_while_placeholderVsequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItem
>sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02@
>sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¯
/sequential_10/lstm_22/while/lstm_cell_22/MatMulMatMulFsequential_10/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ21
/sequential_10/lstm_22/while/lstm_cell_22/MatMul
@sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02B
@sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp
1sequential_10/lstm_22/while/lstm_cell_22/MatMul_1MatMul)sequential_10_lstm_22_while_placeholder_2Hsequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ23
1sequential_10/lstm_22/while/lstm_cell_22/MatMul_1
,sequential_10/lstm_22/while/lstm_cell_22/addAddV29sequential_10/lstm_22/while/lstm_cell_22/MatMul:product:0;sequential_10/lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2.
,sequential_10/lstm_22/while/lstm_cell_22/add
?sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02A
?sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp
0sequential_10/lstm_22/while/lstm_cell_22/BiasAddBiasAdd0sequential_10/lstm_22/while/lstm_cell_22/add:z:0Gsequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ22
0sequential_10/lstm_22/while/lstm_cell_22/BiasAdd¶
8sequential_10/lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_22/while/lstm_cell_22/split/split_dimç
.sequential_10/lstm_22/while/lstm_cell_22/splitSplitAsequential_10/lstm_22/while/lstm_cell_22/split/split_dim:output:09sequential_10/lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split20
.sequential_10/lstm_22/while/lstm_cell_22/splitÛ
0sequential_10/lstm_22/while/lstm_cell_22/SigmoidSigmoid7sequential_10/lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_10/lstm_22/while/lstm_cell_22/Sigmoidß
2sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid7sequential_10/lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_1ù
,sequential_10/lstm_22/while/lstm_cell_22/mulMul6sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_1:y:0)sequential_10_lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_22/while/lstm_cell_22/mulÒ
-sequential_10/lstm_22/while/lstm_cell_22/ReluRelu7sequential_10/lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_10/lstm_22/while/lstm_cell_22/Relu
.sequential_10/lstm_22/while/lstm_cell_22/mul_1Mul4sequential_10/lstm_22/while/lstm_cell_22/Sigmoid:y:0;sequential_10/lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_22/while/lstm_cell_22/mul_1
.sequential_10/lstm_22/while/lstm_cell_22/add_1AddV20sequential_10/lstm_22/while/lstm_cell_22/mul:z:02sequential_10/lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_22/while/lstm_cell_22/add_1ß
2sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid7sequential_10/lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_2Ñ
/sequential_10/lstm_22/while/lstm_cell_22/Relu_1Relu2sequential_10/lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_10/lstm_22/while/lstm_cell_22/Relu_1
.sequential_10/lstm_22/while/lstm_cell_22/mul_2Mul6sequential_10/lstm_22/while/lstm_cell_22/Sigmoid_2:y:0=sequential_10/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_22/while/lstm_cell_22/mul_2Î
@sequential_10/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_22_while_placeholder_1'sequential_10_lstm_22_while_placeholder2sequential_10/lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_22/while/TensorArrayV2Write/TensorListSetItem
!sequential_10/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_22/while/add/yÁ
sequential_10/lstm_22/while/addAddV2'sequential_10_lstm_22_while_placeholder*sequential_10/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_22/while/add
#sequential_10/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_22/while/add_1/yä
!sequential_10/lstm_22/while/add_1AddV2Dsequential_10_lstm_22_while_sequential_10_lstm_22_while_loop_counter,sequential_10/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_22/while/add_1Ã
$sequential_10/lstm_22/while/IdentityIdentity%sequential_10/lstm_22/while/add_1:z:0!^sequential_10/lstm_22/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_22/while/Identityì
&sequential_10/lstm_22/while/Identity_1IdentityJsequential_10_lstm_22_while_sequential_10_lstm_22_while_maximum_iterations!^sequential_10/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_22/while/Identity_1Å
&sequential_10/lstm_22/while/Identity_2Identity#sequential_10/lstm_22/while/add:z:0!^sequential_10/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_22/while/Identity_2ò
&sequential_10/lstm_22/while/Identity_3IdentityPsequential_10/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_22/while/Identity_3æ
&sequential_10/lstm_22/while/Identity_4Identity2sequential_10/lstm_22/while/lstm_cell_22/mul_2:z:0!^sequential_10/lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_22/while/Identity_4æ
&sequential_10/lstm_22/while/Identity_5Identity2sequential_10/lstm_22/while/lstm_cell_22/add_1:z:0!^sequential_10/lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_22/while/Identity_5Ì
 sequential_10/lstm_22/while/NoOpNoOp@^sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp?^sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpA^sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_10/lstm_22/while/NoOp"U
$sequential_10_lstm_22_while_identity-sequential_10/lstm_22/while/Identity:output:0"Y
&sequential_10_lstm_22_while_identity_1/sequential_10/lstm_22/while/Identity_1:output:0"Y
&sequential_10_lstm_22_while_identity_2/sequential_10/lstm_22/while/Identity_2:output:0"Y
&sequential_10_lstm_22_while_identity_3/sequential_10/lstm_22/while/Identity_3:output:0"Y
&sequential_10_lstm_22_while_identity_4/sequential_10/lstm_22/while/Identity_4:output:0"Y
&sequential_10_lstm_22_while_identity_5/sequential_10/lstm_22/while/Identity_5:output:0"
Hsequential_10_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resourceJsequential_10_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"
Isequential_10_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resourceKsequential_10_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"
Gsequential_10_lstm_22_while_lstm_cell_22_matmul_readvariableop_resourceIsequential_10_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"
Asequential_10_lstm_22_while_sequential_10_lstm_22_strided_slice_1Csequential_10_lstm_22_while_sequential_10_lstm_22_strided_slice_1_0"
}sequential_10_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_22_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp?sequential_10/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2
>sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp>sequential_10/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2
@sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp@sequential_10/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
Ã
while_cond_753047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_753047___redundant_placeholder04
0while_while_cond_753047___redundant_placeholder14
0while_while_cond_753047___redundant_placeholder24
0while_while_cond_753047___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


)__inference_dense_25_layer_call_fn_758987

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_7550682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_754041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_754041___redundant_placeholder04
0while_while_cond_754041___redundant_placeholder14
0while_while_cond_754041___redundant_placeholder24
0while_while_cond_754041___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
®?
Ò
while_body_757152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÀU

C__inference_lstm_21_layer_call_and_return_conditional_losses_758165

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758081*
condR
while_cond_758080*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
ø
-__inference_lstm_cell_22_layer_call_fn_759314

inputs
states_0
states_1
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7541742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Õ
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_759045

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®?
Ò
while_body_755661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê

ã
lstm_20_while_cond_756123,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1D
@lstm_20_while_lstm_20_while_cond_756123___redundant_placeholder0D
@lstm_20_while_lstm_20_while_cond_756123___redundant_placeholder1D
@lstm_20_while_lstm_20_while_cond_756123___redundant_placeholder2D
@lstm_20_while_lstm_20_while_cond_756123___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
lstm_20/while/Lessu
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_20/while/Identity"9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÀU

C__inference_lstm_21_layer_call_and_return_conditional_losses_755557

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_755473*
condR
while_cond_755472*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀU

C__inference_lstm_22_layer_call_and_return_conditional_losses_755369

inputs?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_755285*
condR
while_cond_755284*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
¶
(__inference_lstm_20_layer_call_fn_757093

inputs
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7557452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿU

C__inference_lstm_22_layer_call_and_return_conditional_losses_758665
inputs_0?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758581*
condR
while_cond_758580*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ï
û
'sequential_10_lstm_22_while_cond_752625H
Dsequential_10_lstm_22_while_sequential_10_lstm_22_while_loop_counterN
Jsequential_10_lstm_22_while_sequential_10_lstm_22_while_maximum_iterations+
'sequential_10_lstm_22_while_placeholder-
)sequential_10_lstm_22_while_placeholder_1-
)sequential_10_lstm_22_while_placeholder_2-
)sequential_10_lstm_22_while_placeholder_3J
Fsequential_10_lstm_22_while_less_sequential_10_lstm_22_strided_slice_1`
\sequential_10_lstm_22_while_sequential_10_lstm_22_while_cond_752625___redundant_placeholder0`
\sequential_10_lstm_22_while_sequential_10_lstm_22_while_cond_752625___redundant_placeholder1`
\sequential_10_lstm_22_while_sequential_10_lstm_22_while_cond_752625___redundant_placeholder2`
\sequential_10_lstm_22_while_sequential_10_lstm_22_while_cond_752625___redundant_placeholder3(
$sequential_10_lstm_22_while_identity
Þ
 sequential_10/lstm_22/while/LessLess'sequential_10_lstm_22_while_placeholderFsequential_10_lstm_22_while_less_sequential_10_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_22/while/Less
$sequential_10/lstm_22/while/IdentityIdentity$sequential_10/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_22/while/Identity"U
$sequential_10_lstm_22_while_identity-sequential_10/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
õ
Þ
.__inference_sequential_10_layer_call_fn_756065

inputs
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
	unknown_2:
Ø
	unknown_3:
Ø
	unknown_4:	Ø
	unknown_5:
Ø
	unknown_6:
Ø
	unknown_7:	Ø
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_7558262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
ä
I__inference_sequential_10_layer_call_and_return_conditional_losses_755826

inputs!
lstm_20_755790:	Ø"
lstm_20_755792:
Ø
lstm_20_755794:	Ø"
lstm_21_755798:
Ø"
lstm_21_755800:
Ø
lstm_21_755802:	Ø"
lstm_22_755806:
Ø"
lstm_22_755808:
Ø
lstm_22_755810:	Ø#
dense_25_755814:

dense_25_755816:	"
dense_26_755820:	
dense_26_755822:
identity¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢"dropout_35/StatefulPartitionedCall¢"dropout_36/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall¢"dropout_38/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¦
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_755790lstm_20_755792lstm_20_755794*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7557452!
lstm_20/StatefulPartitionedCall
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7555862$
"dropout_35/StatefulPartitionedCallË
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0lstm_21_755798lstm_21_755800lstm_21_755802*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7555572!
lstm_21/StatefulPartitionedCall¾
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_35/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7553982$
"dropout_36/StatefulPartitionedCallË
lstm_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0lstm_22_755806lstm_22_755808lstm_22_755810*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7553692!
lstm_22/StatefulPartitionedCall¾
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7552102$
"dropout_37/StatefulPartitionedCall¾
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_25_755814dense_25_755816*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_7550682"
 dense_25/StatefulPartitionedCall¿
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7551772$
"dropout_38/StatefulPartitionedCall½
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_26_755820dense_26_755822*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_7551112"
 dense_26/StatefulPartitionedCall
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãJ
Ò

lstm_20_while_body_756602,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	ØQ
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØK
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	ØO
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
ØI
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpÓ
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemá
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp÷
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_20/while/lstm_cell_20/MatMulè
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpà
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_20/while/lstm_cell_20/MatMul_1Ø
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_20/while/lstm_cell_20/addà
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpå
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_20/while/lstm_cell_20/BiasAdd
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dim¯
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_20/while/lstm_cell_20/split±
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/Sigmoidµ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_1Á
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/mul¨
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_20/while/lstm_cell_20/ReluÕ
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_1Ê
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/add_1µ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_2§
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/Relu_1Ù
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_2
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1lstm_20_while_placeholder$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_20/while/TensorArrayV2Write/TensorListSetIteml
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add/y
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/addp
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add_1/y
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity¦
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2º
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3®
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_4®
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_5
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_20/while/NoOp"9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ü
¸
(__inference_lstm_20_layer_call_fn_757071
inputs_0
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7531172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ù
Ã
while_cond_758080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758080___redundant_placeholder04
0while_while_cond_758080___redundant_placeholder14
0while_while_cond_758080___redundant_placeholder24
0while_while_cond_758080___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¢1
ë
I__inference_sequential_10_layer_call_and_return_conditional_losses_755964
lstm_20_input!
lstm_20_755928:	Ø"
lstm_20_755930:
Ø
lstm_20_755932:	Ø"
lstm_21_755936:
Ø"
lstm_21_755938:
Ø
lstm_21_755940:	Ø"
lstm_22_755944:
Ø"
lstm_22_755946:
Ø
lstm_22_755948:	Ø#
dense_25_755952:

dense_25_755954:	"
dense_26_755958:	
dense_26_755960:
identity¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢"dropout_35/StatefulPartitionedCall¢"dropout_36/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall¢"dropout_38/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall­
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_755928lstm_20_755930lstm_20_755932*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7557452!
lstm_20/StatefulPartitionedCall
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7555862$
"dropout_35/StatefulPartitionedCallË
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0lstm_21_755936lstm_21_755938lstm_21_755940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7555572!
lstm_21/StatefulPartitionedCall¾
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_35/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7553982$
"dropout_36/StatefulPartitionedCallË
lstm_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0lstm_22_755944lstm_22_755946lstm_22_755948*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7553692!
lstm_22/StatefulPartitionedCall¾
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7552102$
"dropout_37/StatefulPartitionedCall¾
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_25_755952dense_25_755954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_7550682"
 dense_25/StatefulPartitionedCall¿
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7551772$
"dropout_38/StatefulPartitionedCall½
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_26_755958dense_26_755960*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_7551112"
 dense_26/StatefulPartitionedCall
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input


H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_753430

inputs

states
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates


H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_753576

inputs

states
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ù
Ã
while_cond_752845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_752845___redundant_placeholder04
0while_while_cond_752845___redundant_placeholder14
0while_while_cond_752845___redundant_placeholder24
0while_while_cond_752845___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_752978

inputs

states
states_11
matmul_readvariableop_resource:	Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ðÓ
Õ 
"__inference__traced_restore_759699
file_prefix4
 assignvariableop_dense_25_kernel:
/
 assignvariableop_1_dense_25_bias:	5
"assignvariableop_2_dense_26_kernel:	.
 assignvariableop_3_dense_26_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_20_lstm_cell_20_kernel:	ØM
9assignvariableop_10_lstm_20_lstm_cell_20_recurrent_kernel:
Ø<
-assignvariableop_11_lstm_20_lstm_cell_20_bias:	ØC
/assignvariableop_12_lstm_21_lstm_cell_21_kernel:
ØM
9assignvariableop_13_lstm_21_lstm_cell_21_recurrent_kernel:
Ø<
-assignvariableop_14_lstm_21_lstm_cell_21_bias:	ØC
/assignvariableop_15_lstm_22_lstm_cell_22_kernel:
ØM
9assignvariableop_16_lstm_22_lstm_cell_22_recurrent_kernel:
Ø<
-assignvariableop_17_lstm_22_lstm_cell_22_bias:	Ø#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: >
*assignvariableop_22_adam_dense_25_kernel_m:
7
(assignvariableop_23_adam_dense_25_bias_m:	=
*assignvariableop_24_adam_dense_26_kernel_m:	6
(assignvariableop_25_adam_dense_26_bias_m:I
6assignvariableop_26_adam_lstm_20_lstm_cell_20_kernel_m:	ØT
@assignvariableop_27_adam_lstm_20_lstm_cell_20_recurrent_kernel_m:
ØC
4assignvariableop_28_adam_lstm_20_lstm_cell_20_bias_m:	ØJ
6assignvariableop_29_adam_lstm_21_lstm_cell_21_kernel_m:
ØT
@assignvariableop_30_adam_lstm_21_lstm_cell_21_recurrent_kernel_m:
ØC
4assignvariableop_31_adam_lstm_21_lstm_cell_21_bias_m:	ØJ
6assignvariableop_32_adam_lstm_22_lstm_cell_22_kernel_m:
ØT
@assignvariableop_33_adam_lstm_22_lstm_cell_22_recurrent_kernel_m:
ØC
4assignvariableop_34_adam_lstm_22_lstm_cell_22_bias_m:	Ø>
*assignvariableop_35_adam_dense_25_kernel_v:
7
(assignvariableop_36_adam_dense_25_bias_v:	=
*assignvariableop_37_adam_dense_26_kernel_v:	6
(assignvariableop_38_adam_dense_26_bias_v:I
6assignvariableop_39_adam_lstm_20_lstm_cell_20_kernel_v:	ØT
@assignvariableop_40_adam_lstm_20_lstm_cell_20_recurrent_kernel_v:
ØC
4assignvariableop_41_adam_lstm_20_lstm_cell_20_bias_v:	ØJ
6assignvariableop_42_adam_lstm_21_lstm_cell_21_kernel_v:
ØT
@assignvariableop_43_adam_lstm_21_lstm_cell_21_recurrent_kernel_v:
ØC
4assignvariableop_44_adam_lstm_21_lstm_cell_21_bias_v:	ØJ
6assignvariableop_45_adam_lstm_22_lstm_cell_22_kernel_v:
ØT
@assignvariableop_46_adam_lstm_22_lstm_cell_22_recurrent_kernel_v:
ØC
4assignvariableop_47_adam_lstm_22_lstm_cell_22_bias_v:	Ø
identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*¢
valueB1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesð
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices£
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_26_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_20_lstm_cell_20_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Á
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_20_lstm_cell_20_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_20_lstm_cell_20_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_lstm_21_lstm_cell_21_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_lstm_21_lstm_cell_21_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14µ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_21_lstm_cell_21_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_lstm_22_lstm_cell_22_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Á
AssignVariableOp_16AssignVariableOp9assignvariableop_16_lstm_22_lstm_cell_22_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_lstm_22_lstm_cell_22_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_25_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_25_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_26_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_26_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_lstm_20_lstm_cell_20_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27È
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_lstm_20_lstm_cell_20_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¼
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_20_lstm_cell_20_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¾
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_21_lstm_cell_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30È
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_21_lstm_cell_21_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¼
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_21_lstm_cell_21_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¾
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_22_lstm_cell_22_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33È
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_22_lstm_cell_22_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_22_lstm_cell_22_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_25_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_25_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_26_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_26_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_lstm_20_lstm_cell_20_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40È
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_lstm_20_lstm_cell_20_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¼
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_lstm_20_lstm_cell_20_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¾
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_lstm_21_lstm_cell_21_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43È
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_lstm_21_lstm_cell_21_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¼
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_lstm_21_lstm_cell_21_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¾
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_lstm_22_lstm_cell_22_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46È
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_lstm_22_lstm_cell_22_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¼
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_lstm_22_lstm_cell_22_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_479
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpþ
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48f
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_49æ
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
ß
¹
(__inference_lstm_21_layer_call_fn_757714
inputs_0
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7537152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ù
Ã
while_cond_754623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_754623___redundant_placeholder04
0while_while_cond_754623___redundant_placeholder14
0while_while_cond_754623___redundant_placeholder24
0while_while_cond_754623___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ùU

C__inference_lstm_20_layer_call_and_return_conditional_losses_757379
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757295*
condR
while_cond_757294*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Æ
ø
-__inference_lstm_cell_21_layer_call_fn_759199

inputs
states_0
states_1
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7534302
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Õ
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_757692

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759248

inputs
states_0
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
çJ
Ô

lstm_22_while_body_756896,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:
ØQ
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØK
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorM
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:
ØO
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ØI
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpÓ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp÷
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_22/while/lstm_cell_22/MatMulè
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpà
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_22/while/lstm_cell_22/MatMul_1Ø
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_22/while/lstm_cell_22/addà
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpå
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_22/while/lstm_cell_22/BiasAdd
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dim¯
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_22/while/lstm_cell_22/split±
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_22/while/lstm_cell_22/Sigmoidµ
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_22/while/lstm_cell_22/Sigmoid_1Á
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/while/lstm_cell_22/mul¨
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_22/while/lstm_cell_22/ReluÕ
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/mul_1Ê
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/add_1µ
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_22/while/lstm_cell_22/Sigmoid_2§
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_22/while/lstm_cell_22/Relu_1Ù
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/mul_2
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_22/while/TensorArrayV2Write/TensorListSetIteml
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add/y
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/addp
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add_1/y
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity¦
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2º
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3®
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/while/Identity_4®
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/while/Identity_5
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_22/while/NoOp"9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"È
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ß
¹
(__inference_lstm_21_layer_call_fn_757703
inputs_0
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7535132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÀU

C__inference_lstm_21_layer_call_and_return_conditional_losses_758308

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758224*
condR
while_cond_758223*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_36_layer_call_and_return_conditional_losses_758323

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_35_layer_call_and_return_conditional_losses_754721

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_757580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757580___redundant_placeholder04
0while_while_cond_757580___redundant_placeholder14
0while_while_cond_757580___redundant_placeholder24
0while_while_cond_757580___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
²?
Ô
while_body_758081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
Ã
while_cond_753443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_753443___redundant_placeholder04
0while_while_cond_753443___redundant_placeholder14
0while_while_cond_753443___redundant_placeholder24
0while_while_cond_753443___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_754243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_754243___redundant_placeholder04
0while_while_cond_754243___redundant_placeholder14
0while_while_cond_754243___redundant_placeholder24
0while_while_cond_754243___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_757937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757937___redundant_placeholder04
0while_while_cond_757937___redundant_placeholder14
0while_while_cond_757937___redundant_placeholder24
0while_while_cond_757937___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ºU

C__inference_lstm_20_layer_call_and_return_conditional_losses_754708

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_754624*
condR
while_cond_754623*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_758724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
®?
Ò
while_body_754624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ò?

C__inference_lstm_22_layer_call_and_return_conditional_losses_754313

inputs'
lstm_cell_22_754231:
Ø'
lstm_cell_22_754233:
Ø"
lstm_cell_22_754235:	Ø
identity¢$lstm_cell_22/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_754231lstm_cell_22_754233lstm_cell_22_754235*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7541742&
$lstm_cell_22/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_754231lstm_cell_22_754233lstm_cell_22_754235*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_754244*
condR
while_cond_754243*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò?

C__inference_lstm_21_layer_call_and_return_conditional_losses_753513

inputs'
lstm_cell_21_753431:
Ø'
lstm_cell_21_753433:
Ø"
lstm_cell_21_753435:	Ø
identity¢$lstm_cell_21/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_753431lstm_cell_21_753433lstm_cell_21_753435*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7534302&
$lstm_cell_21/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_753431lstm_cell_21_753433lstm_cell_21_753435*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_753444*
condR
while_cond_753443*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
·
(__inference_lstm_22_layer_call_fn_758379

inputs
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7553692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_754028

inputs

states
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ù
Ã
while_cond_758437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758437___redundant_placeholder04
0while_while_cond_758437___redundant_placeholder14
0while_while_cond_758437___redundant_placeholder24
0while_while_cond_758437___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_752832

inputs

states
states_11
matmul_readvariableop_resource:	Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
çJ
Ô

lstm_22_while_body_756404,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:
ØQ
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØK
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorM
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:
ØO
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ØI
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpÓ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp÷
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_22/while/lstm_cell_22/MatMulè
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpà
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_22/while/lstm_cell_22/MatMul_1Ø
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_22/while/lstm_cell_22/addà
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpå
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_22/while/lstm_cell_22/BiasAdd
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dim¯
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_22/while/lstm_cell_22/split±
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_22/while/lstm_cell_22/Sigmoidµ
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_22/while/lstm_cell_22/Sigmoid_1Á
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/while/lstm_cell_22/mul¨
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_22/while/lstm_cell_22/ReluÕ
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/mul_1Ê
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/add_1µ
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_22/while/lstm_cell_22/Sigmoid_2§
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_22/while/lstm_cell_22/Relu_1Ù
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/lstm_cell_22/mul_2
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_22/while/TensorArrayV2Write/TensorListSetIteml
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add/y
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/addp
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add_1/y
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity¦
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2º
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3®
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/while/Identity_4®
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/while/Identity_5
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_22/while/NoOp"9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"È
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ºU

C__inference_lstm_20_layer_call_and_return_conditional_losses_757665

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757581*
condR
while_cond_757580*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_758978

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãJ
Ò

lstm_20_while_body_756124,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	ØQ
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØK
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	ØO
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
ØI
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpÓ
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemá
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp÷
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_20/while/lstm_cell_20/MatMulè
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpà
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_20/while/lstm_cell_20/MatMul_1Ø
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_20/while/lstm_cell_20/addà
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpå
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_20/while/lstm_cell_20/BiasAdd
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dim¯
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_20/while/lstm_cell_20/split±
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/Sigmoidµ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_1Á
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/mul¨
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_20/while/lstm_cell_20/ReluÕ
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_1Ê
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/add_1µ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_2§
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/Relu_1Ù
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_2
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1lstm_20_while_placeholder$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_20/while/TensorArrayV2Write/TensorListSetIteml
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add/y
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/addp
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add_1/y
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity¦
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2º
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3®
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_4®
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_5
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_20/while/NoOp"9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê

ã
lstm_21_while_cond_756748,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1D
@lstm_21_while_lstm_21_while_cond_756748___redundant_placeholder0D
@lstm_21_while_lstm_21_while_cond_756748___redundant_placeholder1D
@lstm_21_while_lstm_21_while_cond_756748___redundant_placeholder2D
@lstm_21_while_lstm_21_while_cond_756748___redundant_placeholder3
lstm_21_while_identity

lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2
lstm_21/while/Lessu
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_21/while/Identity"9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µ
·
(__inference_lstm_22_layer_call_fn_758368

inputs
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7550222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_755284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_755284___redundant_placeholder04
0while_while_cond_755284___redundant_placeholder14
0while_while_cond_755284___redundant_placeholder24
0while_while_cond_755284___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_758723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758723___redundant_placeholder04
0while_while_cond_758723___redundant_placeholder14
0while_while_cond_758723___redundant_placeholder24
0while_while_cond_758723___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ô
G
+__inference_dropout_37_layer_call_fn_758956

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7550352
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
d
+__inference_dropout_37_layer_call_fn_758961

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7552102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_757795
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô
G
+__inference_dropout_35_layer_call_fn_757670

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7547212
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
·
(__inference_lstm_21_layer_call_fn_757736

inputs
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7555572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_755210

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â^

'sequential_10_lstm_20_while_body_752346H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3G
Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0
sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	Ø_
Ksequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØY
Jsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø(
$sequential_10_lstm_20_while_identity*
&sequential_10_lstm_20_while_identity_1*
&sequential_10_lstm_20_while_identity_2*
&sequential_10_lstm_20_while_identity_3*
&sequential_10_lstm_20_while_identity_4*
&sequential_10_lstm_20_while_identity_5E
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	Ø]
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
ØW
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpï
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_20_while_placeholderVsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02@
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¯
/sequential_10/lstm_20/while/lstm_cell_20/MatMulMatMulFsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ21
/sequential_10/lstm_20/while/lstm_cell_20/MatMul
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02B
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1MatMul)sequential_10_lstm_20_while_placeholder_2Hsequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ23
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1
,sequential_10/lstm_20/while/lstm_cell_20/addAddV29sequential_10/lstm_20/while/lstm_cell_20/MatMul:product:0;sequential_10/lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2.
,sequential_10/lstm_20/while/lstm_cell_20/add
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02A
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp
0sequential_10/lstm_20/while/lstm_cell_20/BiasAddBiasAdd0sequential_10/lstm_20/while/lstm_cell_20/add:z:0Gsequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ22
0sequential_10/lstm_20/while/lstm_cell_20/BiasAdd¶
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimç
.sequential_10/lstm_20/while/lstm_cell_20/splitSplitAsequential_10/lstm_20/while/lstm_cell_20/split/split_dim:output:09sequential_10/lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split20
.sequential_10/lstm_20/while/lstm_cell_20/splitÛ
0sequential_10/lstm_20/while/lstm_cell_20/SigmoidSigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_10/lstm_20/while/lstm_cell_20/Sigmoidß
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1ù
,sequential_10/lstm_20/while/lstm_cell_20/mulMul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1:y:0)sequential_10_lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/while/lstm_cell_20/mulÒ
-sequential_10/lstm_20/while/lstm_cell_20/ReluRelu7sequential_10/lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_10/lstm_20/while/lstm_cell_20/Relu
.sequential_10/lstm_20/while/lstm_cell_20/mul_1Mul4sequential_10/lstm_20/while/lstm_cell_20/Sigmoid:y:0;sequential_10/lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/mul_1
.sequential_10/lstm_20/while/lstm_cell_20/add_1AddV20sequential_10/lstm_20/while/lstm_cell_20/mul:z:02sequential_10/lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/add_1ß
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2Ñ
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1Relu2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1
.sequential_10/lstm_20/while/lstm_cell_20/mul_2Mul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2:y:0=sequential_10/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/mul_2Î
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_20_while_placeholder_1'sequential_10_lstm_20_while_placeholder2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItem
!sequential_10/lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_20/while/add/yÁ
sequential_10/lstm_20/while/addAddV2'sequential_10_lstm_20_while_placeholder*sequential_10/lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_20/while/add
#sequential_10/lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_20/while/add_1/yä
!sequential_10/lstm_20/while/add_1AddV2Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counter,sequential_10/lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_20/while/add_1Ã
$sequential_10/lstm_20/while/IdentityIdentity%sequential_10/lstm_20/while/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identityì
&sequential_10/lstm_20/while/Identity_1IdentityJsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_1Å
&sequential_10/lstm_20/while/Identity_2Identity#sequential_10/lstm_20/while/add:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_2ò
&sequential_10/lstm_20/while/Identity_3IdentityPsequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_3æ
&sequential_10/lstm_20/while/Identity_4Identity2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/while/Identity_4æ
&sequential_10/lstm_20/while/Identity_5Identity2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/while/Identity_5Ì
 sequential_10/lstm_20/while/NoOpNoOp@^sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?^sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpA^sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_10/lstm_20/while/NoOp"U
$sequential_10_lstm_20_while_identity-sequential_10/lstm_20/while/Identity:output:0"Y
&sequential_10_lstm_20_while_identity_1/sequential_10/lstm_20/while/Identity_1:output:0"Y
&sequential_10_lstm_20_while_identity_2/sequential_10/lstm_20/while/Identity_2:output:0"Y
&sequential_10_lstm_20_while_identity_3/sequential_10/lstm_20/while/Identity_3:output:0"Y
&sequential_10_lstm_20_while_identity_4/sequential_10/lstm_20/while/Identity_4:output:0"Y
&sequential_10_lstm_20_while_identity_5/sequential_10/lstm_20/while/Identity_5:output:0"
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resourceJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resourceKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resourceIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0"
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Î*
Ð
I__inference_sequential_10_layer_call_and_return_conditional_losses_755118

inputs!
lstm_20_754709:	Ø"
lstm_20_754711:
Ø
lstm_20_754713:	Ø"
lstm_21_754866:
Ø"
lstm_21_754868:
Ø
lstm_21_754870:	Ø"
lstm_22_755023:
Ø"
lstm_22_755025:
Ø
lstm_22_755027:	Ø#
dense_25_755069:

dense_25_755071:	"
dense_26_755112:	
dense_26_755114:
identity¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¦
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_754709lstm_20_754711lstm_20_754713*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7547082!
lstm_20/StatefulPartitionedCall
dropout_35/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7547212
dropout_35/PartitionedCallÃ
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0lstm_21_754866lstm_21_754868lstm_21_754870*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7548652!
lstm_21/StatefulPartitionedCall
dropout_36/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7548782
dropout_36/PartitionedCallÃ
lstm_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0lstm_22_755023lstm_22_755025lstm_22_755027*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7550222!
lstm_22/StatefulPartitionedCall
dropout_37/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7550352
dropout_37/PartitionedCall¶
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_25_755069dense_25_755071*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_7550682"
 dense_25/StatefulPartitionedCall
dropout_38/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7550792
dropout_38/PartitionedCallµ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_26_755112dense_26_755114*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_7551112"
 dense_26/StatefulPartitionedCall
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
û
'sequential_10_lstm_21_while_cond_752485H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3J
Fsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1`
\sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_752485___redundant_placeholder0`
\sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_752485___redundant_placeholder1`
\sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_752485___redundant_placeholder2`
\sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_752485___redundant_placeholder3(
$sequential_10_lstm_21_while_identity
Þ
 sequential_10/lstm_21/while/LessLess'sequential_10_lstm_21_while_placeholderFsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_21/while/Less
$sequential_10/lstm_21/while/IdentityIdentity$sequential_10/lstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identity"U
$sequential_10_lstm_21_while_identity-sequential_10/lstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
²
¶
(__inference_lstm_20_layer_call_fn_757082

inputs
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7547082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_757938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
µ
·
(__inference_lstm_21_layer_call_fn_757725

inputs
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7548652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã*
×
I__inference_sequential_10_layer_call_and_return_conditional_losses_755925
lstm_20_input!
lstm_20_755889:	Ø"
lstm_20_755891:
Ø
lstm_20_755893:	Ø"
lstm_21_755897:
Ø"
lstm_21_755899:
Ø
lstm_21_755901:	Ø"
lstm_22_755905:
Ø"
lstm_22_755907:
Ø
lstm_22_755909:	Ø#
dense_25_755913:

dense_25_755915:	"
dense_26_755919:	
dense_26_755921:
identity¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall­
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_755889lstm_20_755891lstm_20_755893*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7547082!
lstm_20/StatefulPartitionedCall
dropout_35/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7547212
dropout_35/PartitionedCallÃ
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0lstm_21_755897lstm_21_755899lstm_21_755901*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_7548652!
lstm_21/StatefulPartitionedCall
dropout_36/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7548782
dropout_36/PartitionedCallÃ
lstm_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0lstm_22_755905lstm_22_755907lstm_22_755909*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7550222!
lstm_22/StatefulPartitionedCall
dropout_37/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_7550352
dropout_37/PartitionedCall¶
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_25_755913dense_25_755915*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_7550682"
 dense_25/StatefulPartitionedCall
dropout_38/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7550792
dropout_38/PartitionedCallµ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_26_755919dense_26_755921*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_7551112"
 dense_26/StatefulPartitionedCall
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input

d
F__inference_dropout_36_layer_call_and_return_conditional_losses_754878

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_754780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_754780___redundant_placeholder04
0while_while_cond_754780___redundant_placeholder14
0while_while_cond_754780___redundant_placeholder24
0while_while_cond_754780___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ï
û
'sequential_10_lstm_20_while_cond_752345H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3J
Fsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1`
\sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_752345___redundant_placeholder0`
\sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_752345___redundant_placeholder1`
\sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_752345___redundant_placeholder2`
\sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_752345___redundant_placeholder3(
$sequential_10_lstm_20_while_identity
Þ
 sequential_10/lstm_20/while/LessLess'sequential_10_lstm_20_while_placeholderFsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_20/while/Less
$sequential_10/lstm_20/while/IdentityIdentity$sequential_10/lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identity"U
$sequential_10_lstm_20_while_identity-sequential_10/lstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ü
¸
(__inference_lstm_20_layer_call_fn_757060
inputs_0
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_20_layer_call_and_return_conditional_losses_7529152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ã
÷
-__inference_lstm_cell_20_layer_call_fn_759101

inputs
states_0
states_1
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7528322
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1


)__inference_dense_26_layer_call_fn_759054

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_7551112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ%
ç
while_body_754042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_22_754066_0:
Ø/
while_lstm_cell_22_754068_0:
Ø*
while_lstm_cell_22_754070_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_22_754066:
Ø-
while_lstm_cell_22_754068:
Ø(
while_lstm_cell_22_754070:	Ø¢*while/lstm_cell_22/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_754066_0while_lstm_cell_22_754068_0while_lstm_cell_22_754070_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7540282,
*while/lstm_cell_22/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_754066while_lstm_cell_22_754066_0"8
while_lstm_cell_22_754068while_lstm_cell_22_754068_0"8
while_lstm_cell_22_754070while_lstm_cell_22_754070_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
·
d
+__inference_dropout_38_layer_call_fn_759028

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7551772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò?

C__inference_lstm_22_layer_call_and_return_conditional_losses_754111

inputs'
lstm_cell_22_754029:
Ø'
lstm_cell_22_754031:
Ø"
lstm_cell_22_754033:	Ø
identity¢$lstm_cell_22/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_754029lstm_cell_22_754031lstm_cell_22_754033*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7540282&
$lstm_cell_22/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_754029lstm_cell_22_754031lstm_cell_22_754033*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_754042*
condR
while_cond_754041*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_38_layer_call_and_return_conditional_losses_759033

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759378

inputs
states_0
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
í?

C__inference_lstm_20_layer_call_and_return_conditional_losses_753117

inputs&
lstm_cell_20_753035:	Ø'
lstm_cell_20_753037:
Ø"
lstm_cell_20_753039:	Ø
identity¢$lstm_cell_20/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_753035lstm_cell_20_753037lstm_cell_20_753039*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7529782&
$lstm_cell_20/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_753035lstm_cell_20_753037lstm_cell_20_753039*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_753048*
condR
while_cond_753047*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_757294
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757294___redundant_placeholder04
0while_while_cond_757294___redundant_placeholder14
0while_while_cond_757294___redundant_placeholder24
0while_while_cond_757294___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_754937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_754937___redundant_placeholder04
0while_while_cond_754937___redundant_placeholder14
0while_while_cond_754937___redundant_placeholder24
0while_while_cond_754937___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ºU

C__inference_lstm_20_layer_call_and_return_conditional_losses_757522

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757438*
condR
while_cond_757437*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
d
+__inference_dropout_35_layer_call_fn_757675

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_7555862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã%
å
while_body_753048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_20_753072_0:	Ø/
while_lstm_cell_20_753074_0:
Ø*
while_lstm_cell_20_753076_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_20_753072:	Ø-
while_lstm_cell_20_753074:
Ø(
while_lstm_cell_20_753076:	Ø¢*while/lstm_cell_20/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_753072_0while_lstm_cell_20_753074_0while_lstm_cell_20_753076_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7529782,
*while/lstm_cell_20/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_20_753072while_lstm_cell_20_753072_0"8
while_lstm_cell_20_753074while_lstm_cell_20_753074_0"8
while_lstm_cell_20_753076while_lstm_cell_20_753076_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô
G
+__inference_dropout_36_layer_call_fn_758313

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7548782
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_758438
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¼
I__inference_sequential_10_layer_call_and_return_conditional_losses_757049

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	ØI
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
ØC
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	ØG
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ØI
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
ØC
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	ØG
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:
ØI
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ØC
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	Ø>
*dense_25_tensordot_readvariableop_resource:
7
(dense_25_biasadd_readvariableop_resource:	=
*dense_26_tensordot_readvariableop_resource:	6
(dense_26_biasadd_readvariableop_resource:
identity¢dense_25/BiasAdd/ReadVariableOp¢!dense_25/Tensordot/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢!dense_26/Tensordot/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while¢+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢*lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢lstm_21/while¢+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢*lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢lstm_22/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/Shape
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stack
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slices
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/packed/1£
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros/packedo
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros/Const
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zerosw
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/packed/1©
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros_1/packeds
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros_1/Const
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zeros_1
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/perm
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stack
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/TensorArrayV2/element_shapeÒ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2Ï
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensor
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stack
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2¬
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_2Í
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpÍ
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/MatMulÔ
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpÉ
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/MatMul_1À
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/addÌ
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpÍ
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/BiasAdd
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dim
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_20/lstm_cell_20/split
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Sigmoid£
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_1¬
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu½
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_1²
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/add_1£
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_2
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu_1Á
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_2
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_20/TensorArrayV2_1/element_shapeØ
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2_1^
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/time
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counter
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_20_while_body_756602*%
condR
lstm_20_while_cond_756601*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_20/whileÅ
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStack
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_20/strided_slice_3/stack
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2Ë
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_3
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/permÆ
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transpose_1v
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/runtimey
dropout_35/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_35/dropout/Constª
dropout_35/dropout/MulMullstm_20/transpose_1:y:0!dropout_35/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_35/dropout/Mul{
dropout_35/dropout/ShapeShapelstm_20/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_35/dropout/ShapeÚ
/dropout_35/dropout/random_uniform/RandomUniformRandomUniform!dropout_35/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_35/dropout/random_uniform/RandomUniform
!dropout_35/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_35/dropout/GreaterEqual/yï
dropout_35/dropout/GreaterEqualGreaterEqual8dropout_35/dropout/random_uniform/RandomUniform:output:0*dropout_35/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_35/dropout/GreaterEqual¥
dropout_35/dropout/CastCast#dropout_35/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_35/dropout/Cast«
dropout_35/dropout/Mul_1Muldropout_35/dropout/Mul:z:0dropout_35/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_35/dropout/Mul_1j
lstm_21/ShapeShapedropout_35/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_21/Shape
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stack
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slices
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_21/zeros/packed/1£
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros/packedo
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros/Const
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/zerosw
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_21/zeros_1/packed/1©
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros_1/packeds
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros_1/Const
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/zeros_1
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/perm©
lstm_21/transpose	Transposedropout_35/dropout/Mul_1:z:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stack
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_21/TensorArrayV2/element_shapeÒ
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2Ï
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensor
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stack
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2­
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_2Î
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpÍ
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/MatMulÔ
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpÉ
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/MatMul_1À
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/addÌ
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpÍ
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/BiasAdd
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dim
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_21/lstm_cell_21/split
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Sigmoid£
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/lstm_cell_21/Sigmoid_1¬
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Relu½
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul_1²
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/add_1£
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/lstm_cell_21/Sigmoid_2
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Relu_1Á
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul_2
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_21/TensorArrayV2_1/element_shapeØ
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2_1^
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/time
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counter
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_21_while_body_756749*%
condR
lstm_21_while_cond_756748*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_21/whileÅ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStack
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_21/strided_slice_3/stack
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2Ë
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_3
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/permÆ
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transpose_1v
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/runtimey
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_36/dropout/Constª
dropout_36/dropout/MulMullstm_21/transpose_1:y:0!dropout_36/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Mul{
dropout_36/dropout/ShapeShapelstm_21/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_36/dropout/ShapeÚ
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_36/dropout/GreaterEqual/yï
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_36/dropout/GreaterEqual¥
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Cast«
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Mul_1j
lstm_22/ShapeShapedropout_36/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_22/Shape
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slices
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_22/zeros/packed/1£
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/zerosw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_22/zeros_1/packed/1©
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/zeros_1
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/perm©
lstm_22/transpose	Transposedropout_36/dropout/Mul_1:z:0lstm_22/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stack
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_22/TensorArrayV2/element_shapeÒ
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2Ï
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensor
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stack
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2­
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_22/strided_slice_2Î
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpÍ
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/MatMulÔ
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpÉ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/MatMul_1À
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/addÌ
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpÍ
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/BiasAdd
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dim
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_22/lstm_cell_22/split
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Sigmoid£
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/lstm_cell_22/Sigmoid_1¬
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Relu½
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul_1²
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/add_1£
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/lstm_cell_22/Sigmoid_2
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Relu_1Á
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul_2
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_22/TensorArrayV2_1/element_shapeØ
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2_1^
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/time
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counter
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_756896*%
condR
lstm_22_while_cond_756895*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_22/whileÅ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStack
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_22/strided_slice_3/stack
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2Ë
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_22/strided_slice_3
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/permÆ
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/transpose_1v
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/runtimey
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_37/dropout/Constª
dropout_37/dropout/MulMullstm_22/transpose_1:y:0!dropout_37/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Mul{
dropout_37/dropout/ShapeShapelstm_22/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_37/dropout/ShapeÚ
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_37/dropout/GreaterEqual/yï
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_37/dropout/GreaterEqual¥
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Cast«
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Mul_1³
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axes
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedropout_37/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_25/Tensordot/Shape
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axisþ
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axis
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const¤
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1¬
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axisÝ
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat°
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stackÂ
dense_25/Tensordot/transpose	Transposedropout_37/dropout/Mul_1:z:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/transposeÃ
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/ReshapeÃ
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/MatMul
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/Const_2
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axisê
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1µ
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot¨
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp¬
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/BiasAddx
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Reluy
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_38/dropout/Const®
dropout_38/dropout/MulMuldense_25/Relu:activations:0!dropout_38/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_38/dropout/Mul
dropout_38/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_38/dropout/ShapeÚ
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_38/dropout/random_uniform/RandomUniform
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_38/dropout/GreaterEqual/yï
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_38/dropout/GreaterEqual¥
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_38/dropout/Cast«
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_38/dropout/Mul_1²
!dense_26/Tensordot/ReadVariableOpReadVariableOp*dense_26_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_26/Tensordot/ReadVariableOp|
dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/axes
dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_26/Tensordot/free
dense_26/Tensordot/ShapeShapedropout_38/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_26/Tensordot/Shape
 dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/GatherV2/axisþ
dense_26/Tensordot/GatherV2GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/free:output:0)dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2
"dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_26/Tensordot/GatherV2_1/axis
dense_26/Tensordot/GatherV2_1GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/axes:output:0+dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2_1~
dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const¤
dense_26/Tensordot/ProdProd$dense_26/Tensordot/GatherV2:output:0!dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod
dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const_1¬
dense_26/Tensordot/Prod_1Prod&dense_26/Tensordot/GatherV2_1:output:0#dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod_1
dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_26/Tensordot/concat/axisÝ
dense_26/Tensordot/concatConcatV2 dense_26/Tensordot/free:output:0 dense_26/Tensordot/axes:output:0'dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat°
dense_26/Tensordot/stackPack dense_26/Tensordot/Prod:output:0"dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/stackÂ
dense_26/Tensordot/transpose	Transposedropout_38/dropout/Mul_1:z:0"dense_26/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/transposeÃ
dense_26/Tensordot/ReshapeReshape dense_26/Tensordot/transpose:y:0!dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/ReshapeÂ
dense_26/Tensordot/MatMulMatMul#dense_26/Tensordot/Reshape:output:0)dense_26/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/MatMul
dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/Const_2
 dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/concat_1/axisê
dense_26/Tensordot/concat_1ConcatV2$dense_26/Tensordot/GatherV2:output:0#dense_26/Tensordot/Const_2:output:0)dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat_1´
dense_26/TensordotReshape#dense_26/Tensordot/MatMul:product:0$dense_26/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp«
dense_26/BiasAddBiasAdddense_26/Tensordot:output:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/BiasAddx
IdentityIdentitydense_26/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp"^dense_26/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2F
!dense_26/Tensordot/ReadVariableOp!dense_26/Tensordot/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
d
+__inference_dropout_36_layer_call_fn_758318

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_7553982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_758581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

d
F__inference_dropout_38_layer_call_and_return_conditional_losses_755079

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_755398

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë 
ü
D__inference_dense_26_layer_call_and_return_conditional_losses_759084

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²?
Ô
while_body_755285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Õ
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_755177

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ^

'sequential_10_lstm_21_while_body_752486H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3G
Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0
sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
Ø_
Ksequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØY
Jsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø(
$sequential_10_lstm_21_while_identity*
&sequential_10_lstm_21_while_identity_1*
&sequential_10_lstm_21_while_identity_2*
&sequential_10_lstm_21_while_identity_3*
&sequential_10_lstm_21_while_identity_4*
&sequential_10_lstm_21_while_identity_5E
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor[
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
Ø]
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
ØW
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpï
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_21_while_placeholderVsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02@
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¯
/sequential_10/lstm_21/while/lstm_cell_21/MatMulMatMulFsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ21
/sequential_10/lstm_21/while/lstm_cell_21/MatMul
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02B
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1MatMul)sequential_10_lstm_21_while_placeholder_2Hsequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ23
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1
,sequential_10/lstm_21/while/lstm_cell_21/addAddV29sequential_10/lstm_21/while/lstm_cell_21/MatMul:product:0;sequential_10/lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2.
,sequential_10/lstm_21/while/lstm_cell_21/add
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02A
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp
0sequential_10/lstm_21/while/lstm_cell_21/BiasAddBiasAdd0sequential_10/lstm_21/while/lstm_cell_21/add:z:0Gsequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ22
0sequential_10/lstm_21/while/lstm_cell_21/BiasAdd¶
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimç
.sequential_10/lstm_21/while/lstm_cell_21/splitSplitAsequential_10/lstm_21/while/lstm_cell_21/split/split_dim:output:09sequential_10/lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split20
.sequential_10/lstm_21/while/lstm_cell_21/splitÛ
0sequential_10/lstm_21/while/lstm_cell_21/SigmoidSigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_10/lstm_21/while/lstm_cell_21/Sigmoidß
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1ù
,sequential_10/lstm_21/while/lstm_cell_21/mulMul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1:y:0)sequential_10_lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_21/while/lstm_cell_21/mulÒ
-sequential_10/lstm_21/while/lstm_cell_21/ReluRelu7sequential_10/lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_10/lstm_21/while/lstm_cell_21/Relu
.sequential_10/lstm_21/while/lstm_cell_21/mul_1Mul4sequential_10/lstm_21/while/lstm_cell_21/Sigmoid:y:0;sequential_10/lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_21/while/lstm_cell_21/mul_1
.sequential_10/lstm_21/while/lstm_cell_21/add_1AddV20sequential_10/lstm_21/while/lstm_cell_21/mul:z:02sequential_10/lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_21/while/lstm_cell_21/add_1ß
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2Ñ
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1Relu2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1
.sequential_10/lstm_21/while/lstm_cell_21/mul_2Mul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2:y:0=sequential_10/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_21/while/lstm_cell_21/mul_2Î
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_21_while_placeholder_1'sequential_10_lstm_21_while_placeholder2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItem
!sequential_10/lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_21/while/add/yÁ
sequential_10/lstm_21/while/addAddV2'sequential_10_lstm_21_while_placeholder*sequential_10/lstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_21/while/add
#sequential_10/lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_21/while/add_1/yä
!sequential_10/lstm_21/while/add_1AddV2Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counter,sequential_10/lstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_21/while/add_1Ã
$sequential_10/lstm_21/while/IdentityIdentity%sequential_10/lstm_21/while/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identityì
&sequential_10/lstm_21/while/Identity_1IdentityJsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_1Å
&sequential_10/lstm_21/while/Identity_2Identity#sequential_10/lstm_21/while/add:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_2ò
&sequential_10/lstm_21/while/Identity_3IdentityPsequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_3æ
&sequential_10/lstm_21/while/Identity_4Identity2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0!^sequential_10/lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_21/while/Identity_4æ
&sequential_10/lstm_21/while/Identity_5Identity2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_21/while/Identity_5Ì
 sequential_10/lstm_21/while/NoOpNoOp@^sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp?^sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpA^sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_10/lstm_21/while/NoOp"U
$sequential_10_lstm_21_while_identity-sequential_10/lstm_21/while/Identity:output:0"Y
&sequential_10_lstm_21_while_identity_1/sequential_10/lstm_21/while/Identity_1:output:0"Y
&sequential_10_lstm_21_while_identity_2/sequential_10/lstm_21/while/Identity_2:output:0"Y
&sequential_10_lstm_21_while_identity_3/sequential_10/lstm_21/while/Identity_3:output:0"Y
&sequential_10_lstm_21_while_identity_4/sequential_10/lstm_21/while/Identity_4:output:0"Y
&sequential_10_lstm_21_while_identity_5/sequential_10/lstm_21/while/Identity_5:output:0"
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resourceJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resourceKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resourceIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0"
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÿU

C__inference_lstm_21_layer_call_and_return_conditional_losses_757879
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757795*
condR
while_cond_757794*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õî
¼
I__inference_sequential_10_layer_call_and_return_conditional_losses_756543

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	ØI
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
ØC
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	ØG
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ØI
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
ØC
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	ØG
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:
ØI
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ØC
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	Ø>
*dense_25_tensordot_readvariableop_resource:
7
(dense_25_biasadd_readvariableop_resource:	=
*dense_26_tensordot_readvariableop_resource:	6
(dense_26_biasadd_readvariableop_resource:
identity¢dense_25/BiasAdd/ReadVariableOp¢!dense_25/Tensordot/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢!dense_26/Tensordot/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while¢+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢*lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢lstm_21/while¢+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢*lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢lstm_22/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/Shape
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stack
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slices
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/packed/1£
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros/packedo
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros/Const
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zerosw
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/packed/1©
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros_1/packeds
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros_1/Const
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zeros_1
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/perm
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stack
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/TensorArrayV2/element_shapeÒ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2Ï
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensor
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stack
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2¬
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_2Í
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpÍ
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/MatMulÔ
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpÉ
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/MatMul_1À
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/addÌ
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpÍ
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_20/lstm_cell_20/BiasAdd
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dim
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_20/lstm_cell_20/split
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Sigmoid£
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_1¬
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu½
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_1²
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/add_1£
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_2
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu_1Á
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_2
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_20/TensorArrayV2_1/element_shapeØ
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2_1^
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/time
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counter
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_20_while_body_756124*%
condR
lstm_20_while_cond_756123*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_20/whileÅ
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStack
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_20/strided_slice_3/stack
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2Ë
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_3
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/permÆ
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transpose_1v
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/runtime
dropout_35/IdentityIdentitylstm_20/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_35/Identityj
lstm_21/ShapeShapedropout_35/Identity:output:0*
T0*
_output_shapes
:2
lstm_21/Shape
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stack
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slices
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_21/zeros/packed/1£
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros/packedo
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros/Const
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/zerosw
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_21/zeros_1/packed/1©
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros_1/packeds
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros_1/Const
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/zeros_1
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/perm©
lstm_21/transpose	Transposedropout_35/Identity:output:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stack
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_21/TensorArrayV2/element_shapeÒ
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2Ï
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensor
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stack
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2­
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_2Î
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpÍ
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/MatMulÔ
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpÉ
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/MatMul_1À
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/addÌ
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpÍ
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_21/lstm_cell_21/BiasAdd
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dim
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_21/lstm_cell_21/split
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Sigmoid£
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/lstm_cell_21/Sigmoid_1¬
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Relu½
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul_1²
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/add_1£
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/lstm_cell_21/Sigmoid_2
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/Relu_1Á
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/lstm_cell_21/mul_2
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_21/TensorArrayV2_1/element_shapeØ
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2_1^
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/time
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counter
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_21_while_body_756264*%
condR
lstm_21_while_cond_756263*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_21/whileÅ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStack
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_21/strided_slice_3/stack
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2Ë
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_3
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/permÆ
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transpose_1v
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/runtime
dropout_36/IdentityIdentitylstm_21/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/Identityj
lstm_22/ShapeShapedropout_36/Identity:output:0*
T0*
_output_shapes
:2
lstm_22/Shape
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slices
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_22/zeros/packed/1£
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/zerosw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_22/zeros_1/packed/1©
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/zeros_1
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/perm©
lstm_22/transpose	Transposedropout_36/Identity:output:0lstm_22/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stack
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_22/TensorArrayV2/element_shapeÒ
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2Ï
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensor
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stack
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2­
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_22/strided_slice_2Î
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpÍ
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/MatMulÔ
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpÉ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/MatMul_1À
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/addÌ
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpÍ
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_22/lstm_cell_22/BiasAdd
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dim
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_22/lstm_cell_22/split
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Sigmoid£
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/lstm_cell_22/Sigmoid_1¬
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Relu½
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul_1²
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/add_1£
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_22/lstm_cell_22/Sigmoid_2
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/Relu_1Á
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/lstm_cell_22/mul_2
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_22/TensorArrayV2_1/element_shapeØ
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2_1^
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/time
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counter
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_756404*%
condR
lstm_22_while_cond_756403*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_22/whileÅ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStack
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_22/strided_slice_3/stack
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2Ë
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_22/strided_slice_3
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/permÆ
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_22/transpose_1v
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/runtime
dropout_37/IdentityIdentitylstm_22/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/Identity³
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axes
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedropout_37/Identity:output:0*
T0*
_output_shapes
:2
dense_25/Tensordot/Shape
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axisþ
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axis
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const¤
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1¬
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axisÝ
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat°
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stackÂ
dense_25/Tensordot/transpose	Transposedropout_37/Identity:output:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/transposeÃ
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/ReshapeÃ
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot/MatMul
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/Const_2
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axisê
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1µ
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Tensordot¨
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp¬
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/BiasAddx
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/Relu
dropout_38/IdentityIdentitydense_25/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_38/Identity²
!dense_26/Tensordot/ReadVariableOpReadVariableOp*dense_26_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_26/Tensordot/ReadVariableOp|
dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/axes
dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_26/Tensordot/free
dense_26/Tensordot/ShapeShapedropout_38/Identity:output:0*
T0*
_output_shapes
:2
dense_26/Tensordot/Shape
 dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/GatherV2/axisþ
dense_26/Tensordot/GatherV2GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/free:output:0)dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2
"dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_26/Tensordot/GatherV2_1/axis
dense_26/Tensordot/GatherV2_1GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/axes:output:0+dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2_1~
dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const¤
dense_26/Tensordot/ProdProd$dense_26/Tensordot/GatherV2:output:0!dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod
dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const_1¬
dense_26/Tensordot/Prod_1Prod&dense_26/Tensordot/GatherV2_1:output:0#dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod_1
dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_26/Tensordot/concat/axisÝ
dense_26/Tensordot/concatConcatV2 dense_26/Tensordot/free:output:0 dense_26/Tensordot/axes:output:0'dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat°
dense_26/Tensordot/stackPack dense_26/Tensordot/Prod:output:0"dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/stackÂ
dense_26/Tensordot/transpose	Transposedropout_38/Identity:output:0"dense_26/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/transposeÃ
dense_26/Tensordot/ReshapeReshape dense_26/Tensordot/transpose:y:0!dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/ReshapeÂ
dense_26/Tensordot/MatMulMatMul#dense_26/Tensordot/Reshape:output:0)dense_26/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot/MatMul
dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/Const_2
 dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/concat_1/axisê
dense_26/Tensordot/concat_1ConcatV2$dense_26/Tensordot/GatherV2:output:0#dense_26/Tensordot/Const_2:output:0)dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat_1´
dense_26/TensordotReshape#dense_26/Tensordot/MatMul:product:0$dense_26/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/Tensordot§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp«
dense_26/BiasAddBiasAdddense_26/Tensordot:output:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/BiasAddx
IdentityIdentitydense_26/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp"^dense_26/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2F
!dense_26/Tensordot/ReadVariableOp!dense_26/Tensordot/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀU

C__inference_lstm_21_layer_call_and_return_conditional_losses_754865

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_754781*
condR
while_cond_754780*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_35_layer_call_and_return_conditional_losses_757680

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ºU

C__inference_lstm_20_layer_call_and_return_conditional_losses_755745

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_755661*
condR
while_cond_755660*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
.__inference_sequential_10_layer_call_fn_755886
lstm_20_input
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
	unknown_2:
Ø
	unknown_3:
Ø
	unknown_4:	Ø
	unknown_5:
Ø
	unknown_6:
Ø
	unknown_7:	Ø
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_7558262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input
çJ
Ô

lstm_21_while_body_756749,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
ØQ
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØK
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
ØO
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
ØI
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpÓ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp÷
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_21/while/lstm_cell_21/MatMulè
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpà
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_21/while/lstm_cell_21/MatMul_1Ø
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_21/while/lstm_cell_21/addà
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpå
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_21/while/lstm_cell_21/BiasAdd
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dim¯
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_21/while/lstm_cell_21/split±
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_21/while/lstm_cell_21/Sigmoidµ
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_21/while/lstm_cell_21/Sigmoid_1Á
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/while/lstm_cell_21/mul¨
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_21/while/lstm_cell_21/ReluÕ
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/mul_1Ê
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/add_1µ
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_21/while/lstm_cell_21/Sigmoid_2§
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_21/while/lstm_cell_21/Relu_1Ù
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/mul_2
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_21/while/TensorArrayV2Write/TensorListSetIteml
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add/y
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/addp
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add_1/y
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity¦
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2º
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3®
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/while/Identity_4®
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/while/Identity_5
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_21/while/NoOp"9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"È
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÀU

C__inference_lstm_22_layer_call_and_return_conditional_losses_758808

inputs?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758724*
condR
while_cond_758723*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_37_layer_call_and_return_conditional_losses_758966

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_754174

inputs

states
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ô
G
+__inference_dropout_38_layer_call_fn_759023

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_7550792
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®?
Ò
while_body_757438
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Úh
î
__inference__traced_save_759545
file_prefix.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableopD
@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop8
4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop:
6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableopD
@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop8
4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop:
6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableopD
@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop8
4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableopA
=savev2_adam_lstm_20_lstm_cell_20_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_20_lstm_cell_20_bias_m_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableopA
=savev2_adam_lstm_20_lstm_cell_20_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_20_lstm_cell_20_bias_v_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*¢
valueB1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesê
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¬
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableop@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableop@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_m_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_m_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_v_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_v_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*«
_input_shapes
: :
::	:: : : : : :	Ø:
Ø:Ø:
Ø:
Ø:Ø:
Ø:
Ø:Ø: : : : :
::	::	Ø:
Ø:Ø:
Ø:
Ø:Ø:
Ø:
Ø:Ø:
::	::	Ø:
Ø:Ø:
Ø:
Ø:Ø:
Ø:
Ø:Ø: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 
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
:	Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:&"
 
_output_shapes
:
Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:&"
 
_output_shapes
:
Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:
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
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:&"
 
_output_shapes
:
Ø:&"
 
_output_shapes
:
Ø:! 

_output_shapes	
:Ø:&!"
 
_output_shapes
:
Ø:&""
 
_output_shapes
:
Ø:!#

_output_shapes	
:Ø:&$"
 
_output_shapes
:
:!%

_output_shapes	
::%&!

_output_shapes
:	: '

_output_shapes
::%(!

_output_shapes
:	Ø:&)"
 
_output_shapes
:
Ø:!*

_output_shapes	
:Ø:&+"
 
_output_shapes
:
Ø:&,"
 
_output_shapes
:
Ø:!-

_output_shapes	
:Ø:&."
 
_output_shapes
:
Ø:&/"
 
_output_shapes
:
Ø:!0

_output_shapes	
:Ø:1

_output_shapes
: 
²?
Ô
while_body_758867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
Ã
while_cond_757437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757437___redundant_placeholder04
0while_while_cond_757437___redundant_placeholder14
0while_while_cond_757437___redundant_placeholder24
0while_while_cond_757437___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759150

inputs
states_0
states_11
matmul_readvariableop_resource:	Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
²?
Ô
while_body_755473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê

ã
lstm_20_while_cond_756601,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1D
@lstm_20_while_lstm_20_while_cond_756601___redundant_placeholder0D
@lstm_20_while_lstm_20_while_cond_756601___redundant_placeholder1D
@lstm_20_while_lstm_20_while_cond_756601___redundant_placeholder2D
@lstm_20_while_lstm_20_while_cond_756601___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
lstm_20/while/Lessu
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_20/while/Identity"9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÿU

C__inference_lstm_22_layer_call_and_return_conditional_losses_758522
inputs_0?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758438*
condR
while_cond_758437*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
²?
Ô
while_body_754938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_22_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_22_matmul_readvariableop_resource:
ØG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_22_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759182

inputs
states_0
states_11
matmul_readvariableop_resource:	Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ã
÷
-__inference_lstm_cell_20_layer_call_fn_759118

inputs
states_0
states_1
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7529782
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ê

ã
lstm_22_while_cond_756403,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_756403___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_756403___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_756403___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_756403___redundant_placeholder3
lstm_22_while_identity

lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2
lstm_22/while/Lessu
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_22/while/Identity"9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Õ
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_758335

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ùU

C__inference_lstm_20_layer_call_and_return_conditional_losses_757236
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	ØA
-lstm_cell_20_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_20_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757152*
condR
while_cond_757151*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Æ
ø
-__inference_lstm_cell_21_layer_call_fn_759216

inputs
states_0
states_1
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7535762
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ù
Ã
while_cond_758866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758866___redundant_placeholder04
0while_while_cond_758866___redundant_placeholder14
0while_while_cond_758866___redundant_placeholder24
0while_while_cond_758866___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
®?
Ò
while_body_757295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
æ%
ç
while_body_753646
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_21_753670_0:
Ø/
while_lstm_cell_21_753672_0:
Ø*
while_lstm_cell_21_753674_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_21_753670:
Ø-
while_lstm_cell_21_753672:
Ø(
while_lstm_cell_21_753674:	Ø¢*while/lstm_cell_21/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_753670_0while_lstm_cell_21_753672_0while_lstm_cell_21_753674_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7535762,
*while/lstm_cell_21/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_21_753670while_lstm_cell_21_753670_0"8
while_lstm_cell_21_753672while_lstm_cell_21_753672_0"8
while_lstm_cell_21_753674while_lstm_cell_21_753674_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø
Û
$__inference_signature_wrapper_756003
lstm_20_input
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
	unknown_2:
Ø
	unknown_3:
Ø
	unknown_4:	Ø
	unknown_5:
Ø
	unknown_6:
Ø
	unknown_7:	Ø
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_7527652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input
õ
Þ
.__inference_sequential_10_layer_call_fn_756034

inputs
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
	unknown_2:
Ø
	unknown_3:
Ø
	unknown_4:	Ø
	unknown_5:
Ø
	unknown_6:
Ø
	unknown_7:	Ø
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_7551182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ%
ç
while_body_753444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_21_753468_0:
Ø/
while_lstm_cell_21_753470_0:
Ø*
while_lstm_cell_21_753472_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_21_753468:
Ø-
while_lstm_cell_21_753470:
Ø(
while_lstm_cell_21_753472:	Ø¢*while/lstm_cell_21/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_753468_0while_lstm_cell_21_753470_0while_lstm_cell_21_753472_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7534302,
*while/lstm_cell_21/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_21_753468while_lstm_cell_21_753468_0"8
while_lstm_cell_21_753470while_lstm_cell_21_753470_0"8
while_lstm_cell_21_753472while_lstm_cell_21_753472_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ë 
ü
D__inference_dense_26_layer_call_and_return_conditional_losses_755111

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_753645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_753645___redundant_placeholder04
0while_while_cond_753645___redundant_placeholder14
0while_while_cond_753645___redundant_placeholder24
0while_while_cond_753645___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
æ%
ç
while_body_754244
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_22_754268_0:
Ø/
while_lstm_cell_22_754270_0:
Ø*
while_lstm_cell_22_754272_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_22_754268:
Ø-
while_lstm_cell_22_754270:
Ø(
while_lstm_cell_22_754272:	Ø¢*while/lstm_cell_22/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_754268_0while_lstm_cell_22_754270_0while_lstm_cell_22_754272_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_7541742,
*while/lstm_cell_22/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_754268while_lstm_cell_22_754268_0"8
while_lstm_cell_22_754270while_lstm_cell_22_754270_0"8
while_lstm_cell_22_754272while_lstm_cell_22_754272_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

d
F__inference_dropout_37_layer_call_and_return_conditional_losses_755035

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
çJ
Ô

lstm_21_while_body_756264,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
ØQ
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØK
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
ØO
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
ØI
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpÓ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp÷
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2#
!lstm_21/while/lstm_cell_21/MatMulè
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpà
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2%
#lstm_21/while/lstm_cell_21/MatMul_1Ø
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2 
lstm_21/while/lstm_cell_21/addà
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpå
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2$
"lstm_21/while/lstm_cell_21/BiasAdd
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dim¯
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_21/while/lstm_cell_21/split±
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_21/while/lstm_cell_21/Sigmoidµ
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_21/while/lstm_cell_21/Sigmoid_1Á
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_21/while/lstm_cell_21/mul¨
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_21/while/lstm_cell_21/ReluÕ
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/mul_1Ê
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/add_1µ
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_21/while/lstm_cell_21/Sigmoid_2§
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_21/while/lstm_cell_21/Relu_1Ù
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/lstm_cell_21/mul_2
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_21/while/TensorArrayV2Write/TensorListSetIteml
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add/y
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/addp
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add_1/y
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity¦
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2º
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3®
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/while/Identity_4®
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/while/Identity_5
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_21/while/NoOp"9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"È
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ªË
±
!__inference__wrapped_model_752765
lstm_20_inputT
Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource:	ØW
Csequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
ØQ
Bsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	ØU
Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ØW
Csequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
ØQ
Bsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	ØU
Asequential_10_lstm_22_lstm_cell_22_matmul_readvariableop_resource:
ØW
Csequential_10_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ØQ
Bsequential_10_lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	ØL
8sequential_10_dense_25_tensordot_readvariableop_resource:
E
6sequential_10_dense_25_biasadd_readvariableop_resource:	K
8sequential_10_dense_26_tensordot_readvariableop_resource:	D
6sequential_10_dense_26_biasadd_readvariableop_resource:
identity¢-sequential_10/dense_25/BiasAdd/ReadVariableOp¢/sequential_10/dense_25/Tensordot/ReadVariableOp¢-sequential_10/dense_26/BiasAdd/ReadVariableOp¢/sequential_10/dense_26/Tensordot/ReadVariableOp¢9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢sequential_10/lstm_20/while¢9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢sequential_10/lstm_21/while¢9sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢8sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢:sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢sequential_10/lstm_22/whilew
sequential_10/lstm_20/ShapeShapelstm_20_input*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shape 
)sequential_10/lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_20/strided_slice/stack¤
+sequential_10/lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_1¤
+sequential_10/lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_2æ
#sequential_10/lstm_20/strided_sliceStridedSlice$sequential_10/lstm_20/Shape:output:02sequential_10/lstm_20/strided_slice/stack:output:04sequential_10/lstm_20/strided_slice/stack_1:output:04sequential_10/lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_20/strided_slice
$sequential_10/lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2&
$sequential_10/lstm_20/zeros/packed/1Û
"sequential_10/lstm_20/zeros/packedPack,sequential_10/lstm_20/strided_slice:output:0-sequential_10/lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_20/zeros/packed
!sequential_10/lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_20/zeros/ConstÎ
sequential_10/lstm_20/zerosFill+sequential_10/lstm_20/zeros/packed:output:0*sequential_10/lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_20/zeros
&sequential_10/lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&sequential_10/lstm_20/zeros_1/packed/1á
$sequential_10/lstm_20/zeros_1/packedPack,sequential_10/lstm_20/strided_slice:output:0/sequential_10/lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_20/zeros_1/packed
#sequential_10/lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_20/zeros_1/ConstÖ
sequential_10/lstm_20/zeros_1Fill-sequential_10/lstm_20/zeros_1/packed:output:0,sequential_10/lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_20/zeros_1¡
$sequential_10/lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_20/transpose/permÃ
sequential_10/lstm_20/transpose	Transposelstm_20_input-sequential_10/lstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_10/lstm_20/transpose
sequential_10/lstm_20/Shape_1Shape#sequential_10/lstm_20/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shape_1¤
+sequential_10/lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_1/stack¨
-sequential_10/lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_1¨
-sequential_10/lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_2ò
%sequential_10/lstm_20/strided_slice_1StridedSlice&sequential_10/lstm_20/Shape_1:output:04sequential_10/lstm_20/strided_slice_1/stack:output:06sequential_10/lstm_20/strided_slice_1/stack_1:output:06sequential_10/lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_1±
1sequential_10/lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_20/TensorArrayV2/element_shape
#sequential_10/lstm_20/TensorArrayV2TensorListReserve:sequential_10/lstm_20/TensorArrayV2/element_shape:output:0.sequential_10/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_20/TensorArrayV2ë
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_20/transpose:y:0Tsequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor¤
+sequential_10/lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_2/stack¨
-sequential_10/lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_1¨
-sequential_10/lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_2
%sequential_10/lstm_20/strided_slice_2StridedSlice#sequential_10/lstm_20/transpose:y:04sequential_10/lstm_20/strided_slice_2/stack:output:06sequential_10/lstm_20/strided_slice_2/stack_1:output:06sequential_10/lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_2÷
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02:
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp
)sequential_10/lstm_20/lstm_cell_20/MatMulMatMul.sequential_10/lstm_20/strided_slice_2:output:0@sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2+
)sequential_10/lstm_20/lstm_cell_20/MatMulþ
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02<
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp
+sequential_10/lstm_20/lstm_cell_20/MatMul_1MatMul$sequential_10/lstm_20/zeros:output:0Bsequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2-
+sequential_10/lstm_20/lstm_cell_20/MatMul_1ø
&sequential_10/lstm_20/lstm_cell_20/addAddV23sequential_10/lstm_20/lstm_cell_20/MatMul:product:05sequential_10/lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2(
&sequential_10/lstm_20/lstm_cell_20/addö
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02;
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp
*sequential_10/lstm_20/lstm_cell_20/BiasAddBiasAdd*sequential_10/lstm_20/lstm_cell_20/add:z:0Asequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2,
*sequential_10/lstm_20/lstm_cell_20/BiasAddª
2sequential_10/lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_20/lstm_cell_20/split/split_dimÏ
(sequential_10/lstm_20/lstm_cell_20/splitSplit;sequential_10/lstm_20/lstm_cell_20/split/split_dim:output:03sequential_10/lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(sequential_10/lstm_20/lstm_cell_20/splitÉ
*sequential_10/lstm_20/lstm_cell_20/SigmoidSigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/lstm_20/lstm_cell_20/SigmoidÍ
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1ä
&sequential_10/lstm_20/lstm_cell_20/mulMul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_1:y:0&sequential_10/lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/lstm_cell_20/mulÀ
'sequential_10/lstm_20/lstm_cell_20/ReluRelu1sequential_10/lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/lstm_20/lstm_cell_20/Reluõ
(sequential_10/lstm_20/lstm_cell_20/mul_1Mul.sequential_10/lstm_20/lstm_cell_20/Sigmoid:y:05sequential_10/lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/mul_1ê
(sequential_10/lstm_20/lstm_cell_20/add_1AddV2*sequential_10/lstm_20/lstm_cell_20/mul:z:0,sequential_10/lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/add_1Í
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2¿
)sequential_10/lstm_20/lstm_cell_20/Relu_1Relu,sequential_10/lstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_10/lstm_20/lstm_cell_20/Relu_1ù
(sequential_10/lstm_20/lstm_cell_20/mul_2Mul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_2:y:07sequential_10/lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/mul_2»
3sequential_10/lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3sequential_10/lstm_20/TensorArrayV2_1/element_shape
%sequential_10/lstm_20/TensorArrayV2_1TensorListReserve<sequential_10/lstm_20/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_10/lstm_20/TensorArrayV2_1z
sequential_10/lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_10/lstm_20/time«
.sequential_10/lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/maximum_iterations
(sequential_10/lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_20/while/loop_counterÙ
sequential_10/lstm_20/whileWhile1sequential_10/lstm_20/while/loop_counter:output:07sequential_10/lstm_20/while/maximum_iterations:output:0#sequential_10/lstm_20/time:output:0.sequential_10/lstm_20/TensorArrayV2_1:handle:0$sequential_10/lstm_20/zeros:output:0&sequential_10/lstm_20/zeros_1:output:0.sequential_10/lstm_20/strided_slice_1:output:0Msequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resourceCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resourceBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_10_lstm_20_while_body_752346*3
cond+R)
'sequential_10_lstm_20_while_cond_752345*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_10/lstm_20/whileá
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_20/while:output:3Osequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStack­
+sequential_10/lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_20/strided_slice_3/stack¨
-sequential_10/lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_20/strided_slice_3/stack_1¨
-sequential_10/lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_3/stack_2
%sequential_10/lstm_20/strided_slice_3StridedSliceAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_20/strided_slice_3/stack:output:06sequential_10/lstm_20/strided_slice_3/stack_1:output:06sequential_10/lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_3¥
&sequential_10/lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_20/transpose_1/permþ
!sequential_10/lstm_20/transpose_1	TransposeAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/lstm_20/transpose_1
sequential_10/lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_20/runtime°
!sequential_10/dropout_35/IdentityIdentity%sequential_10/lstm_20/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/dropout_35/Identity
sequential_10/lstm_21/ShapeShape*sequential_10/dropout_35/Identity:output:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shape 
)sequential_10/lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_21/strided_slice/stack¤
+sequential_10/lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_1¤
+sequential_10/lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_2æ
#sequential_10/lstm_21/strided_sliceStridedSlice$sequential_10/lstm_21/Shape:output:02sequential_10/lstm_21/strided_slice/stack:output:04sequential_10/lstm_21/strided_slice/stack_1:output:04sequential_10/lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_21/strided_slice
$sequential_10/lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2&
$sequential_10/lstm_21/zeros/packed/1Û
"sequential_10/lstm_21/zeros/packedPack,sequential_10/lstm_21/strided_slice:output:0-sequential_10/lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_21/zeros/packed
!sequential_10/lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_21/zeros/ConstÎ
sequential_10/lstm_21/zerosFill+sequential_10/lstm_21/zeros/packed:output:0*sequential_10/lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_21/zeros
&sequential_10/lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&sequential_10/lstm_21/zeros_1/packed/1á
$sequential_10/lstm_21/zeros_1/packedPack,sequential_10/lstm_21/strided_slice:output:0/sequential_10/lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_21/zeros_1/packed
#sequential_10/lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_21/zeros_1/ConstÖ
sequential_10/lstm_21/zeros_1Fill-sequential_10/lstm_21/zeros_1/packed:output:0,sequential_10/lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_21/zeros_1¡
$sequential_10/lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_21/transpose/permá
sequential_10/lstm_21/transpose	Transpose*sequential_10/dropout_35/Identity:output:0-sequential_10/lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_10/lstm_21/transpose
sequential_10/lstm_21/Shape_1Shape#sequential_10/lstm_21/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shape_1¤
+sequential_10/lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_1/stack¨
-sequential_10/lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_1¨
-sequential_10/lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_2ò
%sequential_10/lstm_21/strided_slice_1StridedSlice&sequential_10/lstm_21/Shape_1:output:04sequential_10/lstm_21/strided_slice_1/stack:output:06sequential_10/lstm_21/strided_slice_1/stack_1:output:06sequential_10/lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_1±
1sequential_10/lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_21/TensorArrayV2/element_shape
#sequential_10/lstm_21/TensorArrayV2TensorListReserve:sequential_10/lstm_21/TensorArrayV2/element_shape:output:0.sequential_10/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_21/TensorArrayV2ë
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_21/transpose:y:0Tsequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor¤
+sequential_10/lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_2/stack¨
-sequential_10/lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_1¨
-sequential_10/lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_2
%sequential_10/lstm_21/strided_slice_2StridedSlice#sequential_10/lstm_21/transpose:y:04sequential_10/lstm_21/strided_slice_2/stack:output:06sequential_10/lstm_21/strided_slice_2/stack_1:output:06sequential_10/lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_2ø
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02:
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp
)sequential_10/lstm_21/lstm_cell_21/MatMulMatMul.sequential_10/lstm_21/strided_slice_2:output:0@sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2+
)sequential_10/lstm_21/lstm_cell_21/MatMulþ
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02<
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp
+sequential_10/lstm_21/lstm_cell_21/MatMul_1MatMul$sequential_10/lstm_21/zeros:output:0Bsequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2-
+sequential_10/lstm_21/lstm_cell_21/MatMul_1ø
&sequential_10/lstm_21/lstm_cell_21/addAddV23sequential_10/lstm_21/lstm_cell_21/MatMul:product:05sequential_10/lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2(
&sequential_10/lstm_21/lstm_cell_21/addö
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02;
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp
*sequential_10/lstm_21/lstm_cell_21/BiasAddBiasAdd*sequential_10/lstm_21/lstm_cell_21/add:z:0Asequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2,
*sequential_10/lstm_21/lstm_cell_21/BiasAddª
2sequential_10/lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_21/lstm_cell_21/split/split_dimÏ
(sequential_10/lstm_21/lstm_cell_21/splitSplit;sequential_10/lstm_21/lstm_cell_21/split/split_dim:output:03sequential_10/lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(sequential_10/lstm_21/lstm_cell_21/splitÉ
*sequential_10/lstm_21/lstm_cell_21/SigmoidSigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/lstm_21/lstm_cell_21/SigmoidÍ
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1ä
&sequential_10/lstm_21/lstm_cell_21/mulMul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_1:y:0&sequential_10/lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_21/lstm_cell_21/mulÀ
'sequential_10/lstm_21/lstm_cell_21/ReluRelu1sequential_10/lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/lstm_21/lstm_cell_21/Reluõ
(sequential_10/lstm_21/lstm_cell_21/mul_1Mul.sequential_10/lstm_21/lstm_cell_21/Sigmoid:y:05sequential_10/lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_21/lstm_cell_21/mul_1ê
(sequential_10/lstm_21/lstm_cell_21/add_1AddV2*sequential_10/lstm_21/lstm_cell_21/mul:z:0,sequential_10/lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_21/lstm_cell_21/add_1Í
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2¿
)sequential_10/lstm_21/lstm_cell_21/Relu_1Relu,sequential_10/lstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_10/lstm_21/lstm_cell_21/Relu_1ù
(sequential_10/lstm_21/lstm_cell_21/mul_2Mul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_2:y:07sequential_10/lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_21/lstm_cell_21/mul_2»
3sequential_10/lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3sequential_10/lstm_21/TensorArrayV2_1/element_shape
%sequential_10/lstm_21/TensorArrayV2_1TensorListReserve<sequential_10/lstm_21/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_10/lstm_21/TensorArrayV2_1z
sequential_10/lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_10/lstm_21/time«
.sequential_10/lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_21/while/maximum_iterations
(sequential_10/lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_21/while/loop_counterÙ
sequential_10/lstm_21/whileWhile1sequential_10/lstm_21/while/loop_counter:output:07sequential_10/lstm_21/while/maximum_iterations:output:0#sequential_10/lstm_21/time:output:0.sequential_10/lstm_21/TensorArrayV2_1:handle:0$sequential_10/lstm_21/zeros:output:0&sequential_10/lstm_21/zeros_1:output:0.sequential_10/lstm_21/strided_slice_1:output:0Msequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resourceCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resourceBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_10_lstm_21_while_body_752486*3
cond+R)
'sequential_10_lstm_21_while_cond_752485*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_10/lstm_21/whileá
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_21/while:output:3Osequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStack­
+sequential_10/lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_21/strided_slice_3/stack¨
-sequential_10/lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_21/strided_slice_3/stack_1¨
-sequential_10/lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_3/stack_2
%sequential_10/lstm_21/strided_slice_3StridedSliceAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_21/strided_slice_3/stack:output:06sequential_10/lstm_21/strided_slice_3/stack_1:output:06sequential_10/lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_3¥
&sequential_10/lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_21/transpose_1/permþ
!sequential_10/lstm_21/transpose_1	TransposeAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/lstm_21/transpose_1
sequential_10/lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_21/runtime°
!sequential_10/dropout_36/IdentityIdentity%sequential_10/lstm_21/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/dropout_36/Identity
sequential_10/lstm_22/ShapeShape*sequential_10/dropout_36/Identity:output:0*
T0*
_output_shapes
:2
sequential_10/lstm_22/Shape 
)sequential_10/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_22/strided_slice/stack¤
+sequential_10/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_22/strided_slice/stack_1¤
+sequential_10/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_22/strided_slice/stack_2æ
#sequential_10/lstm_22/strided_sliceStridedSlice$sequential_10/lstm_22/Shape:output:02sequential_10/lstm_22/strided_slice/stack:output:04sequential_10/lstm_22/strided_slice/stack_1:output:04sequential_10/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_22/strided_slice
$sequential_10/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2&
$sequential_10/lstm_22/zeros/packed/1Û
"sequential_10/lstm_22/zeros/packedPack,sequential_10/lstm_22/strided_slice:output:0-sequential_10/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_22/zeros/packed
!sequential_10/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_22/zeros/ConstÎ
sequential_10/lstm_22/zerosFill+sequential_10/lstm_22/zeros/packed:output:0*sequential_10/lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_22/zeros
&sequential_10/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&sequential_10/lstm_22/zeros_1/packed/1á
$sequential_10/lstm_22/zeros_1/packedPack,sequential_10/lstm_22/strided_slice:output:0/sequential_10/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_22/zeros_1/packed
#sequential_10/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_22/zeros_1/ConstÖ
sequential_10/lstm_22/zeros_1Fill-sequential_10/lstm_22/zeros_1/packed:output:0,sequential_10/lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_22/zeros_1¡
$sequential_10/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_22/transpose/permá
sequential_10/lstm_22/transpose	Transpose*sequential_10/dropout_36/Identity:output:0-sequential_10/lstm_22/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_10/lstm_22/transpose
sequential_10/lstm_22/Shape_1Shape#sequential_10/lstm_22/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_22/Shape_1¤
+sequential_10/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_22/strided_slice_1/stack¨
-sequential_10/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_22/strided_slice_1/stack_1¨
-sequential_10/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_22/strided_slice_1/stack_2ò
%sequential_10/lstm_22/strided_slice_1StridedSlice&sequential_10/lstm_22/Shape_1:output:04sequential_10/lstm_22/strided_slice_1/stack:output:06sequential_10/lstm_22/strided_slice_1/stack_1:output:06sequential_10/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_22/strided_slice_1±
1sequential_10/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_22/TensorArrayV2/element_shape
#sequential_10/lstm_22/TensorArrayV2TensorListReserve:sequential_10/lstm_22/TensorArrayV2/element_shape:output:0.sequential_10/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_22/TensorArrayV2ë
Ksequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_22/transpose:y:0Tsequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensor¤
+sequential_10/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_22/strided_slice_2/stack¨
-sequential_10/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_22/strided_slice_2/stack_1¨
-sequential_10/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_22/strided_slice_2/stack_2
%sequential_10/lstm_22/strided_slice_2StridedSlice#sequential_10/lstm_22/transpose:y:04sequential_10/lstm_22/strided_slice_2/stack:output:06sequential_10/lstm_22/strided_slice_2/stack_1:output:06sequential_10/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_22/strided_slice_2ø
8sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_22_lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02:
8sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp
)sequential_10/lstm_22/lstm_cell_22/MatMulMatMul.sequential_10/lstm_22/strided_slice_2:output:0@sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2+
)sequential_10/lstm_22/lstm_cell_22/MatMulþ
:sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02<
:sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp
+sequential_10/lstm_22/lstm_cell_22/MatMul_1MatMul$sequential_10/lstm_22/zeros:output:0Bsequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2-
+sequential_10/lstm_22/lstm_cell_22/MatMul_1ø
&sequential_10/lstm_22/lstm_cell_22/addAddV23sequential_10/lstm_22/lstm_cell_22/MatMul:product:05sequential_10/lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2(
&sequential_10/lstm_22/lstm_cell_22/addö
9sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02;
9sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp
*sequential_10/lstm_22/lstm_cell_22/BiasAddBiasAdd*sequential_10/lstm_22/lstm_cell_22/add:z:0Asequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2,
*sequential_10/lstm_22/lstm_cell_22/BiasAddª
2sequential_10/lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_22/lstm_cell_22/split/split_dimÏ
(sequential_10/lstm_22/lstm_cell_22/splitSplit;sequential_10/lstm_22/lstm_cell_22/split/split_dim:output:03sequential_10/lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(sequential_10/lstm_22/lstm_cell_22/splitÉ
*sequential_10/lstm_22/lstm_cell_22/SigmoidSigmoid1sequential_10/lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/lstm_22/lstm_cell_22/SigmoidÍ
,sequential_10/lstm_22/lstm_cell_22/Sigmoid_1Sigmoid1sequential_10/lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_22/lstm_cell_22/Sigmoid_1ä
&sequential_10/lstm_22/lstm_cell_22/mulMul0sequential_10/lstm_22/lstm_cell_22/Sigmoid_1:y:0&sequential_10/lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_22/lstm_cell_22/mulÀ
'sequential_10/lstm_22/lstm_cell_22/ReluRelu1sequential_10/lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/lstm_22/lstm_cell_22/Reluõ
(sequential_10/lstm_22/lstm_cell_22/mul_1Mul.sequential_10/lstm_22/lstm_cell_22/Sigmoid:y:05sequential_10/lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_22/lstm_cell_22/mul_1ê
(sequential_10/lstm_22/lstm_cell_22/add_1AddV2*sequential_10/lstm_22/lstm_cell_22/mul:z:0,sequential_10/lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_22/lstm_cell_22/add_1Í
,sequential_10/lstm_22/lstm_cell_22/Sigmoid_2Sigmoid1sequential_10/lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_22/lstm_cell_22/Sigmoid_2¿
)sequential_10/lstm_22/lstm_cell_22/Relu_1Relu,sequential_10/lstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_10/lstm_22/lstm_cell_22/Relu_1ù
(sequential_10/lstm_22/lstm_cell_22/mul_2Mul0sequential_10/lstm_22/lstm_cell_22/Sigmoid_2:y:07sequential_10/lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_22/lstm_cell_22/mul_2»
3sequential_10/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3sequential_10/lstm_22/TensorArrayV2_1/element_shape
%sequential_10/lstm_22/TensorArrayV2_1TensorListReserve<sequential_10/lstm_22/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_10/lstm_22/TensorArrayV2_1z
sequential_10/lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_10/lstm_22/time«
.sequential_10/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_22/while/maximum_iterations
(sequential_10/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_22/while/loop_counterÙ
sequential_10/lstm_22/whileWhile1sequential_10/lstm_22/while/loop_counter:output:07sequential_10/lstm_22/while/maximum_iterations:output:0#sequential_10/lstm_22/time:output:0.sequential_10/lstm_22/TensorArrayV2_1:handle:0$sequential_10/lstm_22/zeros:output:0&sequential_10/lstm_22/zeros_1:output:0.sequential_10/lstm_22/strided_slice_1:output:0Msequential_10/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_22_lstm_cell_22_matmul_readvariableop_resourceCsequential_10_lstm_22_lstm_cell_22_matmul_1_readvariableop_resourceBsequential_10_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_10_lstm_22_while_body_752626*3
cond+R)
'sequential_10_lstm_22_while_cond_752625*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_10/lstm_22/whileá
Fsequential_10/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential_10/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_10/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_22/while:output:3Osequential_10/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential_10/lstm_22/TensorArrayV2Stack/TensorListStack­
+sequential_10/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_22/strided_slice_3/stack¨
-sequential_10/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_22/strided_slice_3/stack_1¨
-sequential_10/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_22/strided_slice_3/stack_2
%sequential_10/lstm_22/strided_slice_3StridedSliceAsequential_10/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_22/strided_slice_3/stack:output:06sequential_10/lstm_22/strided_slice_3/stack_1:output:06sequential_10/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_22/strided_slice_3¥
&sequential_10/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_22/transpose_1/permþ
!sequential_10/lstm_22/transpose_1	TransposeAsequential_10/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/lstm_22/transpose_1
sequential_10/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_22/runtime°
!sequential_10/dropout_37/IdentityIdentity%sequential_10/lstm_22/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/dropout_37/IdentityÝ
/sequential_10/dense_25/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype021
/sequential_10/dense_25/Tensordot/ReadVariableOp
%sequential_10/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_25/Tensordot/axes
%sequential_10/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_10/dense_25/Tensordot/freeª
&sequential_10/dense_25/Tensordot/ShapeShape*sequential_10/dropout_37/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_10/dense_25/Tensordot/Shape¢
.sequential_10/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_25/Tensordot/GatherV2/axisÄ
)sequential_10/dense_25/Tensordot/GatherV2GatherV2/sequential_10/dense_25/Tensordot/Shape:output:0.sequential_10/dense_25/Tensordot/free:output:07sequential_10/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_25/Tensordot/GatherV2¦
0sequential_10/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_25/Tensordot/GatherV2_1/axisÊ
+sequential_10/dense_25/Tensordot/GatherV2_1GatherV2/sequential_10/dense_25/Tensordot/Shape:output:0.sequential_10/dense_25/Tensordot/axes:output:09sequential_10/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_25/Tensordot/GatherV2_1
&sequential_10/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_25/Tensordot/ConstÜ
%sequential_10/dense_25/Tensordot/ProdProd2sequential_10/dense_25/Tensordot/GatherV2:output:0/sequential_10/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_25/Tensordot/Prod
(sequential_10/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_25/Tensordot/Const_1ä
'sequential_10/dense_25/Tensordot/Prod_1Prod4sequential_10/dense_25/Tensordot/GatherV2_1:output:01sequential_10/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_25/Tensordot/Prod_1
,sequential_10/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_25/Tensordot/concat/axis£
'sequential_10/dense_25/Tensordot/concatConcatV2.sequential_10/dense_25/Tensordot/free:output:0.sequential_10/dense_25/Tensordot/axes:output:05sequential_10/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_25/Tensordot/concatè
&sequential_10/dense_25/Tensordot/stackPack.sequential_10/dense_25/Tensordot/Prod:output:00sequential_10/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_25/Tensordot/stackú
*sequential_10/dense_25/Tensordot/transpose	Transpose*sequential_10/dropout_37/Identity:output:00sequential_10/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/dense_25/Tensordot/transposeû
(sequential_10/dense_25/Tensordot/ReshapeReshape.sequential_10/dense_25/Tensordot/transpose:y:0/sequential_10/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_10/dense_25/Tensordot/Reshapeû
'sequential_10/dense_25/Tensordot/MatMulMatMul1sequential_10/dense_25/Tensordot/Reshape:output:07sequential_10/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/dense_25/Tensordot/MatMul
(sequential_10/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_10/dense_25/Tensordot/Const_2¢
.sequential_10/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_25/Tensordot/concat_1/axis°
)sequential_10/dense_25/Tensordot/concat_1ConcatV22sequential_10/dense_25/Tensordot/GatherV2:output:01sequential_10/dense_25/Tensordot/Const_2:output:07sequential_10/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_25/Tensordot/concat_1í
 sequential_10/dense_25/TensordotReshape1sequential_10/dense_25/Tensordot/MatMul:product:02sequential_10/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_10/dense_25/TensordotÒ
-sequential_10/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_10/dense_25/BiasAdd/ReadVariableOpä
sequential_10/dense_25/BiasAddBiasAdd)sequential_10/dense_25/Tensordot:output:05sequential_10/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_25/BiasAdd¢
sequential_10/dense_25/ReluRelu'sequential_10/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_25/Relu´
!sequential_10/dropout_38/IdentityIdentity)sequential_10/dense_25/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/dropout_38/IdentityÜ
/sequential_10/dense_26/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_26_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype021
/sequential_10/dense_26/Tensordot/ReadVariableOp
%sequential_10/dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_26/Tensordot/axes
%sequential_10/dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_10/dense_26/Tensordot/freeª
&sequential_10/dense_26/Tensordot/ShapeShape*sequential_10/dropout_38/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_10/dense_26/Tensordot/Shape¢
.sequential_10/dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_26/Tensordot/GatherV2/axisÄ
)sequential_10/dense_26/Tensordot/GatherV2GatherV2/sequential_10/dense_26/Tensordot/Shape:output:0.sequential_10/dense_26/Tensordot/free:output:07sequential_10/dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_26/Tensordot/GatherV2¦
0sequential_10/dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_26/Tensordot/GatherV2_1/axisÊ
+sequential_10/dense_26/Tensordot/GatherV2_1GatherV2/sequential_10/dense_26/Tensordot/Shape:output:0.sequential_10/dense_26/Tensordot/axes:output:09sequential_10/dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_26/Tensordot/GatherV2_1
&sequential_10/dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_26/Tensordot/ConstÜ
%sequential_10/dense_26/Tensordot/ProdProd2sequential_10/dense_26/Tensordot/GatherV2:output:0/sequential_10/dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_26/Tensordot/Prod
(sequential_10/dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_26/Tensordot/Const_1ä
'sequential_10/dense_26/Tensordot/Prod_1Prod4sequential_10/dense_26/Tensordot/GatherV2_1:output:01sequential_10/dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_26/Tensordot/Prod_1
,sequential_10/dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_26/Tensordot/concat/axis£
'sequential_10/dense_26/Tensordot/concatConcatV2.sequential_10/dense_26/Tensordot/free:output:0.sequential_10/dense_26/Tensordot/axes:output:05sequential_10/dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_26/Tensordot/concatè
&sequential_10/dense_26/Tensordot/stackPack.sequential_10/dense_26/Tensordot/Prod:output:00sequential_10/dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_26/Tensordot/stackú
*sequential_10/dense_26/Tensordot/transpose	Transpose*sequential_10/dropout_38/Identity:output:00sequential_10/dense_26/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/dense_26/Tensordot/transposeû
(sequential_10/dense_26/Tensordot/ReshapeReshape.sequential_10/dense_26/Tensordot/transpose:y:0/sequential_10/dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_10/dense_26/Tensordot/Reshapeú
'sequential_10/dense_26/Tensordot/MatMulMatMul1sequential_10/dense_26/Tensordot/Reshape:output:07sequential_10/dense_26/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/dense_26/Tensordot/MatMul
(sequential_10/dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_10/dense_26/Tensordot/Const_2¢
.sequential_10/dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_26/Tensordot/concat_1/axis°
)sequential_10/dense_26/Tensordot/concat_1ConcatV22sequential_10/dense_26/Tensordot/GatherV2:output:01sequential_10/dense_26/Tensordot/Const_2:output:07sequential_10/dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_26/Tensordot/concat_1ì
 sequential_10/dense_26/TensordotReshape1sequential_10/dense_26/Tensordot/MatMul:product:02sequential_10/dense_26/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_10/dense_26/TensordotÑ
-sequential_10/dense_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_26/BiasAdd/ReadVariableOpã
sequential_10/dense_26/BiasAddBiasAdd)sequential_10/dense_26/Tensordot:output:05sequential_10/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_26/BiasAdd
IdentityIdentity'sequential_10/dense_26/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp.^sequential_10/dense_25/BiasAdd/ReadVariableOp0^sequential_10/dense_25/Tensordot/ReadVariableOp.^sequential_10/dense_26/BiasAdd/ReadVariableOp0^sequential_10/dense_26/Tensordot/ReadVariableOp:^sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9^sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp;^sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^sequential_10/lstm_20/while:^sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp9^sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp;^sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^sequential_10/lstm_21/while:^sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp9^sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp;^sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^sequential_10/lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2^
-sequential_10/dense_25/BiasAdd/ReadVariableOp-sequential_10/dense_25/BiasAdd/ReadVariableOp2b
/sequential_10/dense_25/Tensordot/ReadVariableOp/sequential_10/dense_25/Tensordot/ReadVariableOp2^
-sequential_10/dense_26/BiasAdd/ReadVariableOp-sequential_10/dense_26/BiasAdd/ReadVariableOp2b
/sequential_10/dense_26/Tensordot/ReadVariableOp/sequential_10/dense_26/Tensordot/ReadVariableOp2v
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2t
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp2x
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2:
sequential_10/lstm_20/whilesequential_10/lstm_20/while2v
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2t
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp2x
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2:
sequential_10/lstm_21/whilesequential_10/lstm_21/while2v
9sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp9sequential_10/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2t
8sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp8sequential_10/lstm_22/lstm_cell_22/MatMul/ReadVariableOp2x
:sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:sequential_10/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2:
sequential_10/lstm_22/whilesequential_10/lstm_22/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input

å
.__inference_sequential_10_layer_call_fn_755147
lstm_20_input
unknown:	Ø
	unknown_0:
Ø
	unknown_1:	Ø
	unknown_2:
Ø
	unknown_3:
Ø
	unknown_4:	Ø
	unknown_5:
Ø
	unknown_6:
Ø
	unknown_7:	Ø
	unknown_8:

	unknown_9:	

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_7551182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_20_input


H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759346

inputs
states_0
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ù
Ã
while_cond_755660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_755660___redundant_placeholder04
0while_while_cond_755660___redundant_placeholder14
0while_while_cond_755660___redundant_placeholder24
0while_while_cond_755660___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_755472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_755472___redundant_placeholder04
0while_while_cond_755472___redundant_placeholder14
0while_while_cond_755472___redundant_placeholder24
0while_while_cond_755472___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
²?
Ô
while_body_758224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÀU

C__inference_lstm_22_layer_call_and_return_conditional_losses_758951

inputs?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_758867*
condR
while_cond_758866*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_758223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758223___redundant_placeholder04
0while_while_cond_758223___redundant_placeholder14
0while_while_cond_758223___redundant_placeholder24
0while_while_cond_758223___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
²?
Ô
while_body_754781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ØI
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ØG
3while_lstm_cell_21_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_21_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ß
¹
(__inference_lstm_22_layer_call_fn_758357
inputs_0
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7543132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ò?

C__inference_lstm_21_layer_call_and_return_conditional_losses_753715

inputs'
lstm_cell_21_753633:
Ø'
lstm_cell_21_753635:
Ø"
lstm_cell_21_753637:	Ø
identity¢$lstm_cell_21/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_753633lstm_cell_21_753635lstm_cell_21_753637*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_7535762&
$lstm_cell_21/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_753633lstm_cell_21_753635lstm_cell_21_753637*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_753646*
condR
while_cond_753645*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã%
å
while_body_752846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_20_752870_0:	Ø/
while_lstm_cell_20_752872_0:
Ø*
while_lstm_cell_20_752874_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_20_752870:	Ø-
while_lstm_cell_20_752872:
Ø(
while_lstm_cell_20_752874:	Ø¢*while/lstm_cell_20/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemä
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_752870_0while_lstm_cell_20_752872_0while_lstm_cell_20_752874_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7528322,
*while/lstm_cell_20/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_20_752870while_lstm_cell_20_752870_0"8
while_lstm_cell_20_752872while_lstm_cell_20_752872_0"8
while_lstm_cell_20_752874while_lstm_cell_20_752874_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
Ã
while_cond_757794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757794___redundant_placeholder04
0while_while_cond_757794___redundant_placeholder14
0while_while_cond_757794___redundant_placeholder24
0while_while_cond_757794___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÿU

C__inference_lstm_21_layer_call_and_return_conditional_losses_758022
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:
ØA
-lstm_cell_21_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_21_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_757938*
condR
while_cond_757937*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ê

ã
lstm_21_while_cond_756263,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1D
@lstm_21_while_lstm_21_while_cond_756263___redundant_placeholder0D
@lstm_21_while_lstm_21_while_cond_756263___redundant_placeholder1D
@lstm_21_while_lstm_21_while_cond_756263___redundant_placeholder2D
@lstm_21_while_lstm_21_while_cond_756263___redundant_placeholder3
lstm_21_while_identity

lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2
lstm_21/while/Lessu
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_21/while/Identity"9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Õ!
þ
D__inference_dense_25_layer_call_and_return_conditional_losses_759018

inputs5
!tensordot_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®?
Ò
while_body_757581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	ØI
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
ØC
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	ØG
3while_lstm_cell_20_matmul_1_readvariableop_resource:
ØA
2while_lstm_cell_20_biasadd_readvariableop_resource:	Ø¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	Ø*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759280

inputs
states_0
states_12
matmul_readvariableop_resource:
Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ß
¹
(__inference_lstm_22_layer_call_fn_758346
inputs_0
unknown:
Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_7541112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õ
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_755586

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ã
while_cond_758580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_758580___redundant_placeholder04
0while_while_cond_758580___redundant_placeholder14
0while_while_cond_758580___redundant_placeholder24
0while_while_cond_758580___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
while_cond_757151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_757151___redundant_placeholder04
0while_while_cond_757151___redundant_placeholder14
0while_while_cond_757151___redundant_placeholder24
0while_while_cond_757151___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Õ!
þ
D__inference_dense_25_layer_call_and_return_conditional_losses_755068

inputs5
!tensordot_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

ã
lstm_22_while_cond_756895,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_756895___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_756895___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_756895___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_756895___redundant_placeholder3
lstm_22_while_identity

lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2
lstm_22/while/Lessu
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_22/while/Identity"9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
í?

C__inference_lstm_20_layer_call_and_return_conditional_losses_752915

inputs&
lstm_cell_20_752833:	Ø'
lstm_cell_20_752835:
Ø"
lstm_cell_20_752837:	Ø
identity¢$lstm_cell_20/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2 
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_752833lstm_cell_20_752835lstm_cell_20_752837*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_7528322&
$lstm_cell_20/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÄ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_752833lstm_cell_20_752835lstm_cell_20_752837*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_752846*
condR
while_cond_752845*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀU

C__inference_lstm_22_layer_call_and_return_conditional_losses_755022

inputs?
+lstm_cell_22_matmul_readvariableop_resource:
ØA
-lstm_cell_22_matmul_1_readvariableop_resource:
Ø;
,lstm_cell_22_biasadd_readvariableop_resource:	Ø
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_754938*
condR
while_cond_754937*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
lstm_20_input:
serving_default_lstm_20_input:0ÿÿÿÿÿÿÿÿÿ@
dense_264
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¸
à
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ä__call__
+Å&call_and_return_all_conditional_losses
Æ_default_save_signature"
_tf_keras_sequential
Å
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
regularization_losses
trainable_variables
	variables
	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
 regularization_losses
!trainable_variables
"	variables
#	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
$cell
%
state_spec
&regularization_losses
'trainable_variables
(	variables
)	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
*regularization_losses
+trainable_variables
,	variables
-	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_layer
½

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
§
4regularization_losses
5trainable_variables
6	variables
7	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
½

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
×
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.mª/m«8m¬9m­Cm®Dm¯Em°Fm±Gm²Hm³Im´JmµKm¶.v·/v¸8v¹9vºCv»Dv¼Ev½Fv¾Gv¿HvÀIvÁJvÂKvÃ"
	optimizer
 "
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
Î

Llayers
Mlayer_regularization_losses
regularization_losses
Nmetrics
trainable_variables
	variables
Olayer_metrics
Pnon_trainable_variables
Ä__call__
Æ_default_save_signature
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
-
Ùserving_default"
signature_map
ã
Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
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
¼

Vlayers
Wlayer_regularization_losses

Xstates
regularization_losses
Ymetrics
trainable_variables
	variables
Zlayer_metrics
[non_trainable_variables
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

\layers
]layer_regularization_losses
^metrics
regularization_losses
trainable_variables
	variables
_layer_metrics
`non_trainable_variables
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
ã
a
state_size

Fkernel
Grecurrent_kernel
Hbias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
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
¼

flayers
glayer_regularization_losses

hstates
regularization_losses
imetrics
trainable_variables
	variables
jlayer_metrics
knon_trainable_variables
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

llayers
mlayer_regularization_losses
nmetrics
 regularization_losses
!trainable_variables
"	variables
olayer_metrics
pnon_trainable_variables
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
ã
q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
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
¼

vlayers
wlayer_regularization_losses

xstates
&regularization_losses
ymetrics
'trainable_variables
(	variables
zlayer_metrics
{non_trainable_variables
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±

|layers
}layer_regularization_losses
~metrics
*regularization_losses
+trainable_variables
,	variables
layer_metrics
non_trainable_variables
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_25/kernel
:2dense_25/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
0regularization_losses
1trainable_variables
2	variables
layer_metrics
non_trainable_variables
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
4regularization_losses
5trainable_variables
6	variables
layer_metrics
non_trainable_variables
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_26/kernel
:2dense_26/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
metrics
:regularization_losses
;trainable_variables
<	variables
layer_metrics
non_trainable_variables
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	Ø2lstm_20/lstm_cell_20/kernel
9:7
Ø2%lstm_20/lstm_cell_20/recurrent_kernel
(:&Ø2lstm_20/lstm_cell_20/bias
/:-
Ø2lstm_21/lstm_cell_21/kernel
9:7
Ø2%lstm_21/lstm_cell_21/recurrent_kernel
(:&Ø2lstm_21/lstm_cell_21/bias
/:-
Ø2lstm_22/lstm_cell_22/kernel
9:7
Ø2%lstm_22/lstm_cell_22/recurrent_kernel
(:&Ø2lstm_22/lstm_cell_22/bias
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
 "
trackable_list_wrapper
0
0
1"
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
µ
layers
 layer_regularization_losses
metrics
Rregularization_losses
Strainable_variables
T	variables
layer_metrics
non_trainable_variables
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
µ
layers
 layer_regularization_losses
metrics
bregularization_losses
ctrainable_variables
d	variables
layer_metrics
non_trainable_variables
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
µ
layers
 layer_regularization_losses
metrics
rregularization_losses
strainable_variables
t	variables
layer_metrics
 non_trainable_variables
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
'
$0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

¡total

¢count
£	variables
¤	keras_api"
_tf_keras_metric
c

¥total

¦count
§
_fn_kwargs
¨	variables
©	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
¡0
¢1"
trackable_list_wrapper
.
£	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¥0
¦1"
trackable_list_wrapper
.
¨	variables"
_generic_user_object
(:&
2Adam/dense_25/kernel/m
!:2Adam/dense_25/bias/m
':%	2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
3:1	Ø2"Adam/lstm_20/lstm_cell_20/kernel/m
>:<
Ø2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
-:+Ø2 Adam/lstm_20/lstm_cell_20/bias/m
4:2
Ø2"Adam/lstm_21/lstm_cell_21/kernel/m
>:<
Ø2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
-:+Ø2 Adam/lstm_21/lstm_cell_21/bias/m
4:2
Ø2"Adam/lstm_22/lstm_cell_22/kernel/m
>:<
Ø2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
-:+Ø2 Adam/lstm_22/lstm_cell_22/bias/m
(:&
2Adam/dense_25/kernel/v
!:2Adam/dense_25/bias/v
':%	2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
3:1	Ø2"Adam/lstm_20/lstm_cell_20/kernel/v
>:<
Ø2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
-:+Ø2 Adam/lstm_20/lstm_cell_20/bias/v
4:2
Ø2"Adam/lstm_21/lstm_cell_21/kernel/v
>:<
Ø2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
-:+Ø2 Adam/lstm_21/lstm_cell_21/bias/v
4:2
Ø2"Adam/lstm_22/lstm_cell_22/kernel/v
>:<
Ø2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
-:+Ø2 Adam/lstm_22/lstm_cell_22/bias/v
2
.__inference_sequential_10_layer_call_fn_755147
.__inference_sequential_10_layer_call_fn_756034
.__inference_sequential_10_layer_call_fn_756065
.__inference_sequential_10_layer_call_fn_755886À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_10_layer_call_and_return_conditional_losses_756543
I__inference_sequential_10_layer_call_and_return_conditional_losses_757049
I__inference_sequential_10_layer_call_and_return_conditional_losses_755925
I__inference_sequential_10_layer_call_and_return_conditional_losses_755964À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÒBÏ
!__inference__wrapped_model_752765lstm_20_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_lstm_20_layer_call_fn_757060
(__inference_lstm_20_layer_call_fn_757071
(__inference_lstm_20_layer_call_fn_757082
(__inference_lstm_20_layer_call_fn_757093Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_20_layer_call_and_return_conditional_losses_757236
C__inference_lstm_20_layer_call_and_return_conditional_losses_757379
C__inference_lstm_20_layer_call_and_return_conditional_losses_757522
C__inference_lstm_20_layer_call_and_return_conditional_losses_757665Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_35_layer_call_fn_757670
+__inference_dropout_35_layer_call_fn_757675´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_35_layer_call_and_return_conditional_losses_757680
F__inference_dropout_35_layer_call_and_return_conditional_losses_757692´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_lstm_21_layer_call_fn_757703
(__inference_lstm_21_layer_call_fn_757714
(__inference_lstm_21_layer_call_fn_757725
(__inference_lstm_21_layer_call_fn_757736Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_21_layer_call_and_return_conditional_losses_757879
C__inference_lstm_21_layer_call_and_return_conditional_losses_758022
C__inference_lstm_21_layer_call_and_return_conditional_losses_758165
C__inference_lstm_21_layer_call_and_return_conditional_losses_758308Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_36_layer_call_fn_758313
+__inference_dropout_36_layer_call_fn_758318´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_36_layer_call_and_return_conditional_losses_758323
F__inference_dropout_36_layer_call_and_return_conditional_losses_758335´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_lstm_22_layer_call_fn_758346
(__inference_lstm_22_layer_call_fn_758357
(__inference_lstm_22_layer_call_fn_758368
(__inference_lstm_22_layer_call_fn_758379Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_22_layer_call_and_return_conditional_losses_758522
C__inference_lstm_22_layer_call_and_return_conditional_losses_758665
C__inference_lstm_22_layer_call_and_return_conditional_losses_758808
C__inference_lstm_22_layer_call_and_return_conditional_losses_758951Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_37_layer_call_fn_758956
+__inference_dropout_37_layer_call_fn_758961´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_37_layer_call_and_return_conditional_losses_758966
F__inference_dropout_37_layer_call_and_return_conditional_losses_758978´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_25_layer_call_fn_758987¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_25_layer_call_and_return_conditional_losses_759018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_38_layer_call_fn_759023
+__inference_dropout_38_layer_call_fn_759028´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_38_layer_call_and_return_conditional_losses_759033
F__inference_dropout_38_layer_call_and_return_conditional_losses_759045´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_26_layer_call_fn_759054¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_26_layer_call_and_return_conditional_losses_759084¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÑBÎ
$__inference_signature_wrapper_756003lstm_20_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢2
-__inference_lstm_cell_20_layer_call_fn_759101
-__inference_lstm_cell_20_layer_call_fn_759118¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759150
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759182¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
-__inference_lstm_cell_21_layer_call_fn_759199
-__inference_lstm_cell_21_layer_call_fn_759216¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759248
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759280¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
-__inference_lstm_cell_22_layer_call_fn_759297
-__inference_lstm_cell_22_layer_call_fn_759314¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759346
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759378¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ª
!__inference__wrapped_model_752765CDEFGHIJK./89:¢7
0¢-
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2
dense_26&#
dense_26ÿÿÿÿÿÿÿÿÿ®
D__inference_dense_25_layer_call_and_return_conditional_losses_759018f./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_25_layer_call_fn_758987Y./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
D__inference_dense_26_layer_call_and_return_conditional_losses_759084e894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_26_layer_call_fn_759054X894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_dropout_35_layer_call_and_return_conditional_losses_757680f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_dropout_35_layer_call_and_return_conditional_losses_757692f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_35_layer_call_fn_757670Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_35_layer_call_fn_757675Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_dropout_36_layer_call_and_return_conditional_losses_758323f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_dropout_36_layer_call_and_return_conditional_losses_758335f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_36_layer_call_fn_758313Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_36_layer_call_fn_758318Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_dropout_37_layer_call_and_return_conditional_losses_758966f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_dropout_37_layer_call_and_return_conditional_losses_758978f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_37_layer_call_fn_758956Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_37_layer_call_fn_758961Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_dropout_38_layer_call_and_return_conditional_losses_759033f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_dropout_38_layer_call_and_return_conditional_losses_759045f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_38_layer_call_fn_759023Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_38_layer_call_fn_759028Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÓ
C__inference_lstm_20_layer_call_and_return_conditional_losses_757236CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ó
C__inference_lstm_20_layer_call_and_return_conditional_losses_757379CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
C__inference_lstm_20_layer_call_and_return_conditional_losses_757522rCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¹
C__inference_lstm_20_layer_call_and_return_conditional_losses_757665rCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ª
(__inference_lstm_20_layer_call_fn_757060~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
(__inference_lstm_20_layer_call_fn_757071~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_lstm_20_layer_call_fn_757082eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_20_layer_call_fn_757093eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
C__inference_lstm_21_layer_call_and_return_conditional_losses_757879FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
C__inference_lstm_21_layer_call_and_return_conditional_losses_758022FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
C__inference_lstm_21_layer_call_and_return_conditional_losses_758165sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 º
C__inference_lstm_21_layer_call_and_return_conditional_losses_758308sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 «
(__inference_lstm_21_layer_call_fn_757703FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(__inference_lstm_21_layer_call_fn_757714FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_lstm_21_layer_call_fn_757725fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_21_layer_call_fn_757736fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
C__inference_lstm_22_layer_call_and_return_conditional_losses_758522IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
C__inference_lstm_22_layer_call_and_return_conditional_losses_758665IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
C__inference_lstm_22_layer_call_and_return_conditional_losses_758808sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 º
C__inference_lstm_22_layer_call_and_return_conditional_losses_758951sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 «
(__inference_lstm_22_layer_call_fn_758346IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(__inference_lstm_22_layer_call_fn_758357IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_lstm_22_layer_call_fn_758368fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_22_layer_call_fn_758379fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÏ
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759150CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ï
H__inference_lstm_cell_20_layer_call_and_return_conditional_losses_759182CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¤
-__inference_lstm_cell_20_layer_call_fn_759101òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¤
-__inference_lstm_cell_20_layer_call_fn_759118òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÑ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759248FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ñ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_759280FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¦
-__inference_lstm_cell_21_layer_call_fn_759199ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¦
-__inference_lstm_cell_21_layer_call_fn_759216ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÑ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759346IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ñ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_759378IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¦
-__inference_lstm_cell_22_layer_call_fn_759297ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¦
-__inference_lstm_cell_22_layer_call_fn_759314ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿË
I__inference_sequential_10_layer_call_and_return_conditional_losses_755925~CDEFGHIJK./89B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ë
I__inference_sequential_10_layer_call_and_return_conditional_losses_755964~CDEFGHIJK./89B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ä
I__inference_sequential_10_layer_call_and_return_conditional_losses_756543wCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ä
I__inference_sequential_10_layer_call_and_return_conditional_losses_757049wCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 £
.__inference_sequential_10_layer_call_fn_755147qCDEFGHIJK./89B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
.__inference_sequential_10_layer_call_fn_755886qCDEFGHIJK./89B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_10_layer_call_fn_756034jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_10_layer_call_fn_756065jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¾
$__inference_signature_wrapper_756003CDEFGHIJK./89K¢H
¢ 
Aª>
<
lstm_20_input+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ"7ª4
2
dense_26&#
dense_26ÿÿÿÿÿÿÿÿÿ