¡ß
£ù
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68Â

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
: *
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
: *
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0
|
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 * 
shared_namedense_40/kernel
u
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel* 
_output_shapes
:
 *
dtype0
s
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_40/bias
l
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes	
:*
dtype0
|
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_41/kernel
u
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel* 
_output_shapes
:
*
dtype0
s
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
l
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes	
:*
dtype0
|
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_42/kernel
u
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel* 
_output_shapes
:
*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:*
dtype0
{
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_43/kernel
t
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel*
_output_shapes
:	*
dtype0
r
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_43/bias
k
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes
:*
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

Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_20/kernel/m

+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_21/kernel/m

+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/m
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes
: *
dtype0

Adam/dense_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *'
shared_nameAdam/dense_40/kernel/m

*Adam/dense_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/m* 
_output_shapes
:
 *
dtype0

Adam/dense_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_40/bias/m
z
(Adam/dense_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_41/kernel/m

*Adam/dense_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_41/bias/m
z
(Adam/dense_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_42/kernel/m

*Adam/dense_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_42/bias/m
z
(Adam/dense_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_43/kernel/m

*Adam/dense_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_43/bias/m
y
(Adam/dense_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_20/kernel/v

+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_21/kernel/v

+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/v
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes
: *
dtype0

Adam/dense_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *'
shared_nameAdam/dense_40/kernel/v

*Adam/dense_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/v* 
_output_shapes
:
 *
dtype0

Adam/dense_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_40/bias/v
z
(Adam/dense_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_41/kernel/v

*Adam/dense_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_41/bias/v
z
(Adam/dense_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_42/kernel/v

*Adam/dense_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_42/bias/v
z
(Adam/dense_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_43/kernel/v

*Adam/dense_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_43/bias/v
y
(Adam/dense_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ç\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¢\
value\B\ B\
ê
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
¦

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
¦

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
¦

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
¦

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
´
\iter

]beta_1

^beta_2
	_decay
`learning_ratem§m¨"m©#mª<m«=m¬Dm­Em®Lm¯Mm°Tm±Um²v³v´"vµ#v¶<v·=v¸Dv¹EvºLv»Mv¼Tv½Uv¾*
Z
0
1
"2
#3
<4
=5
D6
E7
L8
M9
T10
U11*
Z
0
1
"2
#3
<4
=5
D6
E7
L8
M9
T10
U11*

a0
b1
c2* 
°
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

iserving_default* 
`Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_40/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
	
a0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_41/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
	
b0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_42/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
	
c0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_43/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

T0
U1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
J
0
1
2
3
4
5
6
7
	8

9*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
a0* 
* 
* 
* 
* 
	
b0* 
* 
* 
* 
* 
	
c0* 
* 
* 
* 
* 
* 
* 
<

total

count
 	variables
¡	keras_api*
M

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

 	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¢0
£1*

¥	variables*
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_40/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_40/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_41/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_41/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_42/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_42/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_43/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_43/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_40/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_40/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_41/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_41/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_42/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_42/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_43/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_43/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_20_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ@@

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_20_inputconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_145905
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¥
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOp#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp*Adam/dense_40/kernel/m/Read/ReadVariableOp(Adam/dense_40/bias/m/Read/ReadVariableOp*Adam/dense_41/kernel/m/Read/ReadVariableOp(Adam/dense_41/bias/m/Read/ReadVariableOp*Adam/dense_42/kernel/m/Read/ReadVariableOp(Adam/dense_42/bias/m/Read/ReadVariableOp*Adam/dense_43/kernel/m/Read/ReadVariableOp(Adam/dense_43/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp*Adam/dense_40/kernel/v/Read/ReadVariableOp(Adam/dense_40/bias/v/Read/ReadVariableOp*Adam/dense_41/kernel/v/Read/ReadVariableOp(Adam/dense_41/bias/v/Read/ReadVariableOp*Adam/dense_42/kernel/v/Read/ReadVariableOp(Adam/dense_42/bias/v/Read/ReadVariableOp*Adam/dense_43/kernel/v/Read/ReadVariableOp(Adam/dense_43/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
__inference__traced_save_146306
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/dense_40/kernel/mAdam/dense_40/bias/mAdam/dense_41/kernel/mAdam/dense_41/bias/mAdam/dense_42/kernel/mAdam/dense_42/bias/mAdam/dense_43/kernel/mAdam/dense_43/bias/mAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/dense_40/kernel/vAdam/dense_40/bias/vAdam/dense_41/kernel/vAdam/dense_41/bias/vAdam/dense_42/kernel/vAdam/dense_42/bias/vAdam/dense_43/kernel/vAdam/dense_43/bias/v*9
Tin2
02.*
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
"__inference__traced_restore_146451º	

Ä
.__inference_sequential_10_layer_call_fn_145701

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_145300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¼
¬
D__inference_dense_41_layer_call_and_return_conditional_losses_146063

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
M
1__inference_max_pooling2d_10_layer_call_fn_145930

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\
¸
__inference__traced_save_146306
file_prefix/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop5
1savev2_adam_dense_40_kernel_m_read_readvariableop3
/savev2_adam_dense_40_bias_m_read_readvariableop5
1savev2_adam_dense_41_kernel_m_read_readvariableop3
/savev2_adam_dense_41_bias_m_read_readvariableop5
1savev2_adam_dense_42_kernel_m_read_readvariableop3
/savev2_adam_dense_42_bias_m_read_readvariableop5
1savev2_adam_dense_43_kernel_m_read_readvariableop3
/savev2_adam_dense_43_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop5
1savev2_adam_dense_40_kernel_v_read_readvariableop3
/savev2_adam_dense_40_bias_v_read_readvariableop5
1savev2_adam_dense_41_kernel_v_read_readvariableop3
/savev2_adam_dense_41_bias_v_read_readvariableop5
1savev2_adam_dense_42_kernel_v_read_readvariableop3
/savev2_adam_dense_42_bias_v_read_readvariableop5
1savev2_adam_dense_43_kernel_v_read_readvariableop3
/savev2_adam_dense_43_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: £
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ì
valueÂB¿.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B õ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop1savev2_adam_dense_40_kernel_m_read_readvariableop/savev2_adam_dense_40_bias_m_read_readvariableop1savev2_adam_dense_41_kernel_m_read_readvariableop/savev2_adam_dense_41_bias_m_read_readvariableop1savev2_adam_dense_42_kernel_m_read_readvariableop/savev2_adam_dense_42_bias_m_read_readvariableop1savev2_adam_dense_43_kernel_m_read_readvariableop/savev2_adam_dense_43_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop1savev2_adam_dense_40_kernel_v_read_readvariableop/savev2_adam_dense_40_bias_v_read_readvariableop1savev2_adam_dense_41_kernel_v_read_readvariableop/savev2_adam_dense_41_bias_v_read_readvariableop1savev2_adam_dense_42_kernel_v_read_readvariableop/savev2_adam_dense_42_bias_v_read_readvariableop1savev2_adam_dense_43_kernel_v_read_readvariableop/savev2_adam_dense_43_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : : :  : :
 ::
::
::	:: : : : : : : : : : : :  : :
 ::
::
::	:: : :  : :
 ::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :&&"
 
_output_shapes
:
 :!'

_output_shapes	
::&("
 
_output_shapes
:
:!)

_output_shapes	
::&*"
 
_output_shapes
:
:!+

_output_shapes	
::%,!

_output_shapes
:	: -

_output_shapes
::.

_output_shapes
: 
ì

*__inference_conv2d_20_layer_call_fn_145914

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145935

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã
$__inference_signature_wrapper_145905
conv2d_20_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_145094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input
±

b
F__inference_flatten_10_layer_call_and_return_conditional_losses_145999

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Y
ê

I__inference_sequential_10_layer_call_and_return_conditional_losses_145874

inputsB
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: ;
'dense_40_matmul_readvariableop_resource:
 7
(dense_40_biasadd_readvariableop_resource:	;
'dense_41_matmul_readvariableop_resource:
7
(dense_41_biasadd_readvariableop_resource:	;
'dense_42_matmul_readvariableop_resource:
7
(dense_42_biasadd_readvariableop_resource:	:
'dense_43_matmul_readvariableop_resource:	6
(dense_43_biasadd_readvariableop_resource:
identity¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢dense_42/BiasAdd/ReadVariableOp¢dense_42/MatMul/ReadVariableOp¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢dense_43/BiasAdd/ReadVariableOp¢dense_43/MatMul/ReadVariableOp
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0­
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ ®
max_pooling2d_10/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0È
conv2d_21/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ »
average_pooling2d_10/AvgPoolAvgPoolconv2d_21/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
g
up_sampling2d_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_10/mulMulup_sampling2d_10/Const:output:0!up_sampling2d_10/Const_1:output:0*
T0*
_output_shapes
:Û
-up_sampling2d_10/resize/ResizeNearestNeighborResizeNearestNeighbor%average_pooling2d_10/AvgPool:output:0up_sampling2d_10/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   «
flatten_10/ReshapeReshape>up_sampling2d_10/resize/ResizeNearestNeighbor:resized_images:0flatten_10/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_40/MatMulMatMulflatten_10/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_41/MatMulMatMuldense_40/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_42/MatMulMatMuldense_41/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_43/SoftmaxSoftmaxdense_43/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_43/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
½
³
__inference_loss_fn_2_146148N
:dense_42_kernel_regularizer_square_readvariableop_resource:

identity¢1dense_42/kernel/Regularizer/Square/ReadVariableOp®
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_42_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_42/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp

Ä
.__inference_sequential_10_layer_call_fn_145730

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_145480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
±

b
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤

ö
D__inference_dense_43_layer_call_and_return_conditional_losses_145275

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
l
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
¬
D__inference_dense_42_layer_call_and_return_conditional_losses_145258

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_42/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145925

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Y
ê

I__inference_sequential_10_layer_call_and_return_conditional_losses_145802

inputsB
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: ;
'dense_40_matmul_readvariableop_resource:
 7
(dense_40_biasadd_readvariableop_resource:	;
'dense_41_matmul_readvariableop_resource:
7
(dense_41_biasadd_readvariableop_resource:	;
'dense_42_matmul_readvariableop_resource:
7
(dense_42_biasadd_readvariableop_resource:	:
'dense_43_matmul_readvariableop_resource:	6
(dense_43_biasadd_readvariableop_resource:
identity¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢dense_42/BiasAdd/ReadVariableOp¢dense_42/MatMul/ReadVariableOp¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢dense_43/BiasAdd/ReadVariableOp¢dense_43/MatMul/ReadVariableOp
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0­
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ ®
max_pooling2d_10/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0È
conv2d_21/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ »
average_pooling2d_10/AvgPoolAvgPoolconv2d_21/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
g
up_sampling2d_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_10/mulMulup_sampling2d_10/Const:output:0!up_sampling2d_10/Const_1:output:0*
T0*
_output_shapes
:Û
-up_sampling2d_10/resize/ResizeNearestNeighborResizeNearestNeighbor%average_pooling2d_10/AvgPool:output:0up_sampling2d_10/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   «
flatten_10/ReshapeReshape>up_sampling2d_10/resize/ResizeNearestNeighbor:resized_images:0flatten_10/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_40/MatMulMatMulflatten_10/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_41/MatMulMatMuldense_40/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_42/MatMulMatMuldense_41/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_43/SoftmaxSoftmaxdense_43/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_43/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
É

)__inference_dense_42_layer_call_fn_146078

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_42_layer_call_and_return_conditional_losses_145258p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
·
Í
.__inference_sequential_10_layer_call_fn_145327
conv2d_20_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_145300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input
¡
l
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145965

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
Q
5__inference_average_pooling2d_10_layer_call_fn_145960

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E

I__inference_sequential_10_layer_call_and_return_conditional_losses_145300

inputs*
conv2d_20_145156: 
conv2d_20_145158: *
conv2d_21_145174:  
conv2d_21_145176: #
dense_40_145213:
 
dense_40_145215:	#
dense_41_145236:

dense_41_145238:	#
dense_42_145259:

dense_42_145261:	"
dense_43_145276:	
dense_43_145278:
identity¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢ dense_42/StatefulPartitionedCall¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢ dense_43/StatefulPartitionedCallü
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_20_145156conv2d_20_145158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155ô
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_21_145174conv2d_21_145176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173ü
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115
 up_sampling2d_10/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134è
flatten_10/PartitionedCallPartitionedCall)up_sampling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_40_145213dense_40_145215*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_145212
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_145236dense_41_145238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_145235
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_145259dense_42_145261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_42_layer_call_and_return_conditional_losses_145258
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_145276dense_43_145278*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_145275
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_145213* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_145236* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_145259* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¼
¬
D__inference_dense_41_layer_call_and_return_conditional_losses_145235

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å

)__inference_dense_43_layer_call_fn_146104

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_145275o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

)__inference_dense_41_layer_call_fn_146046

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_145235p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
¬
D__inference_dense_40_layer_call_and_return_conditional_losses_146031

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë³
Ô
"__inference__traced_restore_146451
file_prefix;
!assignvariableop_conv2d_20_kernel: /
!assignvariableop_1_conv2d_20_bias: =
#assignvariableop_2_conv2d_21_kernel:  /
!assignvariableop_3_conv2d_21_bias: 6
"assignvariableop_4_dense_40_kernel:
 /
 assignvariableop_5_dense_40_bias:	6
"assignvariableop_6_dense_41_kernel:
/
 assignvariableop_7_dense_41_bias:	6
"assignvariableop_8_dense_42_kernel:
/
 assignvariableop_9_dense_42_bias:	6
#assignvariableop_10_dense_43_kernel:	/
!assignvariableop_11_dense_43_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: E
+assignvariableop_21_adam_conv2d_20_kernel_m: 7
)assignvariableop_22_adam_conv2d_20_bias_m: E
+assignvariableop_23_adam_conv2d_21_kernel_m:  7
)assignvariableop_24_adam_conv2d_21_bias_m: >
*assignvariableop_25_adam_dense_40_kernel_m:
 7
(assignvariableop_26_adam_dense_40_bias_m:	>
*assignvariableop_27_adam_dense_41_kernel_m:
7
(assignvariableop_28_adam_dense_41_bias_m:	>
*assignvariableop_29_adam_dense_42_kernel_m:
7
(assignvariableop_30_adam_dense_42_bias_m:	=
*assignvariableop_31_adam_dense_43_kernel_m:	6
(assignvariableop_32_adam_dense_43_bias_m:E
+assignvariableop_33_adam_conv2d_20_kernel_v: 7
)assignvariableop_34_adam_conv2d_20_bias_v: E
+assignvariableop_35_adam_conv2d_21_kernel_v:  7
)assignvariableop_36_adam_conv2d_21_bias_v: >
*assignvariableop_37_adam_dense_40_kernel_v:
 7
(assignvariableop_38_adam_dense_40_bias_v:	>
*assignvariableop_39_adam_dense_41_kernel_v:
7
(assignvariableop_40_adam_dense_41_bias_v:	>
*assignvariableop_41_adam_dense_42_kernel_v:
7
(assignvariableop_42_adam_dense_42_bias_v:	=
*assignvariableop_43_adam_dense_43_kernel_v:	6
(assignvariableop_44_adam_dense_43_bias_v:
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¦
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ì
valueÂB¿.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÌ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_40_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_40_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_41_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_41_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_42_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_42_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_43_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_43_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_20_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_20_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_21_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_21_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_40_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_40_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_41_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_41_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_42_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_42_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_43_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_43_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_20_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_20_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_21_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_21_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_40_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_40_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_41_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_41_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_42_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_42_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_43_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_43_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442(
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
E

I__inference_sequential_10_layer_call_and_return_conditional_losses_145592
conv2d_20_input*
conv2d_20_145539: 
conv2d_20_145541: *
conv2d_21_145545:  
conv2d_21_145547: #
dense_40_145553:
 
dense_40_145555:	#
dense_41_145558:

dense_41_145560:	#
dense_42_145563:

dense_42_145565:	"
dense_43_145568:	
dense_43_145570:
identity¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢ dense_42/StatefulPartitionedCall¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢ dense_43/StatefulPartitionedCall
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputconv2d_20_145539conv2d_20_145541*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155ô
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_21_145545conv2d_21_145547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173ü
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115
 up_sampling2d_10/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134è
flatten_10/PartitionedCallPartitionedCall)up_sampling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_40_145553dense_40_145555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_145212
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_145558dense_41_145560*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_145235
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_145563dense_42_145565*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_42_layer_call_and_return_conditional_losses_145258
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_145568dense_43_145570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_145275
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_145553* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_145558* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_145563* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input

þ
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145955

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ù

)__inference_dense_40_layer_call_fn_146014

inputs
unknown:
 
	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_145212p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
M
1__inference_up_sampling2d_10_layer_call_fn_145970

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E

I__inference_sequential_10_layer_call_and_return_conditional_losses_145648
conv2d_20_input*
conv2d_20_145595: 
conv2d_20_145597: *
conv2d_21_145601:  
conv2d_21_145603: #
dense_40_145609:
 
dense_40_145611:	#
dense_41_145614:

dense_41_145616:	#
dense_42_145619:

dense_42_145621:	"
dense_43_145624:	
dense_43_145626:
identity¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢ dense_42/StatefulPartitionedCall¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢ dense_43/StatefulPartitionedCall
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputconv2d_20_145595conv2d_20_145597*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155ô
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_21_145601conv2d_21_145603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173ü
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115
 up_sampling2d_10/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134è
flatten_10/PartitionedCallPartitionedCall)up_sampling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_40_145609dense_40_145611*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_145212
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_145614dense_41_145616*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_145235
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_145619dense_42_145621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_42_layer_call_and_return_conditional_losses_145258
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_145624dense_43_145626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_145275
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_145609* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_145614* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_145619* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input
¼
¬
D__inference_dense_42_layer_call_and_return_conditional_losses_146095

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_42/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

*__inference_conv2d_21_layer_call_fn_145944

inputs!
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤

ö
D__inference_dense_43_layer_call_and_return_conditional_losses_146115

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÈP
ÿ
!__inference__wrapped_model_145094
conv2d_20_inputP
6sequential_10_conv2d_20_conv2d_readvariableop_resource: E
7sequential_10_conv2d_20_biasadd_readvariableop_resource: P
6sequential_10_conv2d_21_conv2d_readvariableop_resource:  E
7sequential_10_conv2d_21_biasadd_readvariableop_resource: I
5sequential_10_dense_40_matmul_readvariableop_resource:
 E
6sequential_10_dense_40_biasadd_readvariableop_resource:	I
5sequential_10_dense_41_matmul_readvariableop_resource:
E
6sequential_10_dense_41_biasadd_readvariableop_resource:	I
5sequential_10_dense_42_matmul_readvariableop_resource:
E
6sequential_10_dense_42_biasadd_readvariableop_resource:	H
5sequential_10_dense_43_matmul_readvariableop_resource:	D
6sequential_10_dense_43_biasadd_readvariableop_resource:
identity¢.sequential_10/conv2d_20/BiasAdd/ReadVariableOp¢-sequential_10/conv2d_20/Conv2D/ReadVariableOp¢.sequential_10/conv2d_21/BiasAdd/ReadVariableOp¢-sequential_10/conv2d_21/Conv2D/ReadVariableOp¢-sequential_10/dense_40/BiasAdd/ReadVariableOp¢,sequential_10/dense_40/MatMul/ReadVariableOp¢-sequential_10/dense_41/BiasAdd/ReadVariableOp¢,sequential_10/dense_41/MatMul/ReadVariableOp¢-sequential_10/dense_42/BiasAdd/ReadVariableOp¢,sequential_10/dense_42/MatMul/ReadVariableOp¢-sequential_10/dense_43/BiasAdd/ReadVariableOp¢,sequential_10/dense_43/MatMul/ReadVariableOp¬
-sequential_10/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ò
sequential_10/conv2d_20/Conv2DConv2Dconv2d_20_input5sequential_10/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
paddingSAME*
strides
¢
.sequential_10/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Å
sequential_10/conv2d_20/BiasAddBiasAdd'sequential_10/conv2d_20/Conv2D:output:06sequential_10/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ 
sequential_10/conv2d_20/ReluRelu(sequential_10/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ Ê
&sequential_10/max_pooling2d_10/MaxPoolMaxPool*sequential_10/conv2d_20/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
¬
-sequential_10/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ò
sequential_10/conv2d_21/Conv2DConv2D/sequential_10/max_pooling2d_10/MaxPool:output:05sequential_10/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¢
.sequential_10/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Å
sequential_10/conv2d_21/BiasAddBiasAdd'sequential_10/conv2d_21/Conv2D:output:06sequential_10/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential_10/conv2d_21/ReluRelu(sequential_10/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ×
*sequential_10/average_pooling2d_10/AvgPoolAvgPool*sequential_10/conv2d_21/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
u
$sequential_10/up_sampling2d_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"      w
&sequential_10/up_sampling2d_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_10/up_sampling2d_10/mulMul-sequential_10/up_sampling2d_10/Const:output:0/sequential_10/up_sampling2d_10/Const_1:output:0*
T0*
_output_shapes
:
;sequential_10/up_sampling2d_10/resize/ResizeNearestNeighborResizeNearestNeighbor3sequential_10/average_pooling2d_10/AvgPool:output:0&sequential_10/up_sampling2d_10/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(o
sequential_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Õ
 sequential_10/flatten_10/ReshapeReshapeLsequential_10/up_sampling2d_10/resize/ResizeNearestNeighbor:resized_images:0'sequential_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
,sequential_10/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_40_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0»
sequential_10/dense_40/MatMulMatMul)sequential_10/flatten_10/Reshape:output:04sequential_10/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential_10/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential_10/dense_40/BiasAddBiasAdd'sequential_10/dense_40/MatMul:product:05sequential_10/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_10/dense_40/ReluRelu'sequential_10/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,sequential_10/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0»
sequential_10/dense_41/MatMulMatMul)sequential_10/dense_40/Relu:activations:04sequential_10/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential_10/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential_10/dense_41/BiasAddBiasAdd'sequential_10/dense_41/MatMul:product:05sequential_10/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_10/dense_41/ReluRelu'sequential_10/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,sequential_10/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_42_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0»
sequential_10/dense_42/MatMulMatMul)sequential_10/dense_41/Relu:activations:04sequential_10/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential_10/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential_10/dense_42/BiasAddBiasAdd'sequential_10/dense_42/MatMul:product:05sequential_10/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_10/dense_42/ReluRelu'sequential_10/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_10/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_43_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0º
sequential_10/dense_43/MatMulMatMul)sequential_10/dense_42/Relu:activations:04sequential_10/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_10/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_10/dense_43/BiasAddBiasAdd'sequential_10/dense_43/MatMul:product:05sequential_10/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_10/dense_43/SoftmaxSoftmax'sequential_10/dense_43/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_10/dense_43/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^sequential_10/conv2d_20/BiasAdd/ReadVariableOp.^sequential_10/conv2d_20/Conv2D/ReadVariableOp/^sequential_10/conv2d_21/BiasAdd/ReadVariableOp.^sequential_10/conv2d_21/Conv2D/ReadVariableOp.^sequential_10/dense_40/BiasAdd/ReadVariableOp-^sequential_10/dense_40/MatMul/ReadVariableOp.^sequential_10/dense_41/BiasAdd/ReadVariableOp-^sequential_10/dense_41/MatMul/ReadVariableOp.^sequential_10/dense_42/BiasAdd/ReadVariableOp-^sequential_10/dense_42/MatMul/ReadVariableOp.^sequential_10/dense_43/BiasAdd/ReadVariableOp-^sequential_10/dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2`
.sequential_10/conv2d_20/BiasAdd/ReadVariableOp.sequential_10/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_10/conv2d_20/Conv2D/ReadVariableOp-sequential_10/conv2d_20/Conv2D/ReadVariableOp2`
.sequential_10/conv2d_21/BiasAdd/ReadVariableOp.sequential_10/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_10/conv2d_21/Conv2D/ReadVariableOp-sequential_10/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_10/dense_40/BiasAdd/ReadVariableOp-sequential_10/dense_40/BiasAdd/ReadVariableOp2\
,sequential_10/dense_40/MatMul/ReadVariableOp,sequential_10/dense_40/MatMul/ReadVariableOp2^
-sequential_10/dense_41/BiasAdd/ReadVariableOp-sequential_10/dense_41/BiasAdd/ReadVariableOp2\
,sequential_10/dense_41/MatMul/ReadVariableOp,sequential_10/dense_41/MatMul/ReadVariableOp2^
-sequential_10/dense_42/BiasAdd/ReadVariableOp-sequential_10/dense_42/BiasAdd/ReadVariableOp2\
,sequential_10/dense_42/MatMul/ReadVariableOp,sequential_10/dense_42/MatMul/ReadVariableOp2^
-sequential_10/dense_43/BiasAdd/ReadVariableOp-sequential_10/dense_43/BiasAdd/ReadVariableOp2\
,sequential_10/dense_43/MatMul/ReadVariableOp,sequential_10/dense_43/MatMul/ReadVariableOp:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input
½
³
__inference_loss_fn_0_146126N
:dense_40_kernel_regularizer_square_readvariableop_resource:
 
identity¢1dense_40/kernel/Regularizer/Square/ReadVariableOp®
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_40_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_40/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp
½
³
__inference_loss_fn_1_146137N
:dense_41_kernel_regularizer_square_readvariableop_resource:

identity¢1dense_41/kernel/Regularizer/Square/ReadVariableOp®
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_41_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_41/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp

h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
¬
D__inference_dense_40_layer_call_and_return_conditional_losses_145212

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
Í
.__inference_sequential_10_layer_call_fn_145536
conv2d_20_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_145480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
)
_user_specified_nameconv2d_20_input
E

I__inference_sequential_10_layer_call_and_return_conditional_losses_145480

inputs*
conv2d_20_145427: 
conv2d_20_145429: *
conv2d_21_145433:  
conv2d_21_145435: #
dense_40_145441:
 
dense_40_145443:	#
dense_41_145446:

dense_41_145448:	#
dense_42_145451:

dense_42_145453:	"
dense_43_145456:	
dense_43_145458:
identity¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢ dense_42/StatefulPartitionedCall¢1dense_42/kernel/Regularizer/Square/ReadVariableOp¢ dense_43/StatefulPartitionedCallü
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_20_145427conv2d_20_145429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145155ô
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145103
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_21_145433conv2d_21_145435*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173ü
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145115
 up_sampling2d_10/PartitionedCallPartitionedCall-average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145134è
flatten_10/PartitionedCallPartitionedCall)up_sampling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_40_145441dense_40_145443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_145212
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_145446dense_41_145448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_145235
 dense_42/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_42_145451dense_42_145453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_42_layer_call_and_return_conditional_losses_145258
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_145456dense_43_145458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_145275
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_145441* 
_output_shapes
:
 *
dtype0
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 r
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_145446* 
_output_shapes
:
*
dtype0
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_145451* 
_output_shapes
:
*
dtype0
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
r
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@@: : : : : : : : : : : : 2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

þ
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145173

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç
G
+__inference_flatten_10_layer_call_fn_145987

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_145193i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

h
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145982

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ã
serving_default¯
S
conv2d_20_input@
!serving_default_conv2d_20_input:0ÿÿÿÿÿÿÿÿÿ@@<
dense_430
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¼¸

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
»

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
\iter

]beta_1

^beta_2
	_decay
`learning_ratem§m¨"m©#mª<m«=m¬Dm­Em®Lm¯Mm°Tm±Um²v³v´"vµ#v¶<v·=v¸Dv¹EvºLv»Mv¼Tv½Uv¾"
	optimizer
v
0
1
"2
#3
<4
=5
D6
E7
L8
M9
T10
U11"
trackable_list_wrapper
v
0
1
"2
#3
<4
=5
D6
E7
L8
M9
T10
U11"
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
Ê
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_10_layer_call_fn_145327
.__inference_sequential_10_layer_call_fn_145701
.__inference_sequential_10_layer_call_fn_145730
.__inference_sequential_10_layer_call_fn_145536À
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
I__inference_sequential_10_layer_call_and_return_conditional_losses_145802
I__inference_sequential_10_layer_call_and_return_conditional_losses_145874
I__inference_sequential_10_layer_call_and_return_conditional_losses_145592
I__inference_sequential_10_layer_call_and_return_conditional_losses_145648À
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
ÔBÑ
!__inference__wrapped_model_145094conv2d_20_input"
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
,
iserving_default"
signature_map
*:( 2conv2d_20/kernel
: 2conv2d_20/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_20_layer_call_fn_145914¢
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
ï2ì
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145925¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_10_layer_call_fn_145930¢
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
ö2ó
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145935¢
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
*:(  2conv2d_21/kernel
: 2conv2d_21/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_21_layer_call_fn_145944¢
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
ï2ì
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145955¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_average_pooling2d_10_layer_call_fn_145960¢
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
ú2÷
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145965¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_up_sampling2d_10_layer_call_fn_145970¢
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
ö2ó
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145982¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_10_layer_call_fn_145987¢
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
ð2í
F__inference_flatten_10_layer_call_and_return_conditional_losses_145999¢
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
#:!
 2dense_40/kernel
:2dense_40/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
'
a0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_40_layer_call_fn_146014¢
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
D__inference_dense_40_layer_call_and_return_conditional_losses_146031¢
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
#:!
2dense_41/kernel
:2dense_41/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_41_layer_call_fn_146046¢
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
D__inference_dense_41_layer_call_and_return_conditional_losses_146063¢
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
#:!
2dense_42/kernel
:2dense_42/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_42_layer_call_fn_146078¢
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
D__inference_dense_42_layer_call_and_return_conditional_losses_146095¢
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
": 	2dense_43/kernel
:2dense_43/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_43_layer_call_fn_146104¢
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
D__inference_dense_43_layer_call_and_return_conditional_losses_146115¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
³2°
__inference_loss_fn_0_146126
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_146137
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_146148
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
$__inference_signature_wrapper_145905conv2d_20_input"
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
'
a0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
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
R

total

count
 	variables
¡	keras_api"
_tf_keras_metric
c

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
.
¥	variables"
_generic_user_object
/:- 2Adam/conv2d_20/kernel/m
!: 2Adam/conv2d_20/bias/m
/:-  2Adam/conv2d_21/kernel/m
!: 2Adam/conv2d_21/bias/m
(:&
 2Adam/dense_40/kernel/m
!:2Adam/dense_40/bias/m
(:&
2Adam/dense_41/kernel/m
!:2Adam/dense_41/bias/m
(:&
2Adam/dense_42/kernel/m
!:2Adam/dense_42/bias/m
':%	2Adam/dense_43/kernel/m
 :2Adam/dense_43/bias/m
/:- 2Adam/conv2d_20/kernel/v
!: 2Adam/conv2d_20/bias/v
/:-  2Adam/conv2d_21/kernel/v
!: 2Adam/conv2d_21/bias/v
(:&
 2Adam/dense_40/kernel/v
!:2Adam/dense_40/bias/v
(:&
2Adam/dense_41/kernel/v
!:2Adam/dense_41/bias/v
(:&
2Adam/dense_42/kernel/v
!:2Adam/dense_42/bias/v
':%	2Adam/dense_43/kernel/v
 :2Adam/dense_43/bias/v«
!__inference__wrapped_model_145094"#<=DELMTU@¢=
6¢3
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@
ª "3ª0
.
dense_43"
dense_43ÿÿÿÿÿÿÿÿÿó
P__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_145965R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
5__inference_average_pooling2d_10_layer_call_fn_145960R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_20_layer_call_and_return_conditional_losses_145925l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@ 
 
*__inference_conv2d_20_layer_call_fn_145914_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª " ÿÿÿÿÿÿÿÿÿ@@ µ
E__inference_conv2d_21_layer_call_and_return_conditional_losses_145955l"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_21_layer_call_fn_145944_"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ®
D__inference_dense_40_layer_call_and_return_conditional_losses_146031f<=8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_40_layer_call_fn_146014Y<=8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_41_layer_call_and_return_conditional_losses_146063^DE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_41_layer_call_fn_146046QDE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_42_layer_call_and_return_conditional_losses_146095^LM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_42_layer_call_fn_146078QLM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_43_layer_call_and_return_conditional_losses_146115]TU0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_43_layer_call_fn_146104PTU0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÅ
F__inference_flatten_10_layer_call_and_return_conditional_losses_145999{I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_10_layer_call_fn_145987nI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_146126<¢

¢ 
ª " ;
__inference_loss_fn_1_146137D¢

¢ 
ª " ;
__inference_loss_fn_2_146148L¢

¢ 
ª " ï
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_145935R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_10_layer_call_fn_145930R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
I__inference_sequential_10_layer_call_and_return_conditional_losses_145592"#<=DELMTUH¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
I__inference_sequential_10_layer_call_and_return_conditional_losses_145648"#<=DELMTUH¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
I__inference_sequential_10_layer_call_and_return_conditional_losses_145802v"#<=DELMTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
I__inference_sequential_10_layer_call_and_return_conditional_losses_145874v"#<=DELMTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
.__inference_sequential_10_layer_call_fn_145327r"#<=DELMTUH¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
.__inference_sequential_10_layer_call_fn_145536r"#<=DELMTUH¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_10_layer_call_fn_145701i"#<=DELMTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_10_layer_call_fn_145730i"#<=DELMTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "ÿÿÿÿÿÿÿÿÿÁ
$__inference_signature_wrapper_145905"#<=DELMTUS¢P
¢ 
IªF
D
conv2d_20_input1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ@@"3ª0
.
dense_43"
dense_43ÿÿÿÿÿÿÿÿÿï
L__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_145982R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_up_sampling2d_10_layer_call_fn_145970R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ