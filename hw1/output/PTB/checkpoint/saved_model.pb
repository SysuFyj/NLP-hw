��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��2*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
��2*
dtype0
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:02*'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:02*
dtype0
�
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:I2*'
shared_nameembedding_2/embeddings
�
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes

:I2*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dJ*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:dJ*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:J*
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
�
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��2*,
shared_nameAdam/embedding/embeddings/m
�
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m* 
_output_shapes
:
��2*
dtype0
�
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:02*.
shared_nameAdam/embedding_1/embeddings/m
�
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:02*
dtype0
�
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:I2*.
shared_nameAdam/embedding_2/embeddings/m
�
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*
_output_shapes

:I2*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�d*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dJ*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:dJ*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:J*
dtype0
�
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��2*,
shared_nameAdam/embedding/embeddings/v
�
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v* 
_output_shapes
:
��2*
dtype0
�
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:02*.
shared_nameAdam/embedding_1/embeddings/v
�
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:02*
dtype0
�
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:I2*.
shared_nameAdam/embedding_2/embeddings/v
�
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*
_output_shapes

:I2*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�d*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dJ*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:dJ*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:J*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�9 B�9
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
 
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m�/m�0m�=m�>m�v�v�v�/v�0v�=v�>v�
1
0
1
2
/3
04
=5
>6
1
0
1
2
/3
04
=5
>6
 
�

Hlayers
trainable_variables
Inon_trainable_variables
	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
regularization_losses
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

Mlayers
trainable_variables
Nnon_trainable_variables
	variables
Ometrics
Player_metrics
Qlayer_regularization_losses
regularization_losses
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

Rlayers
trainable_variables
Snon_trainable_variables
	variables
Tmetrics
Ulayer_metrics
Vlayer_regularization_losses
regularization_losses
fd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

Wlayers
trainable_variables
Xnon_trainable_variables
 	variables
Ymetrics
Zlayer_metrics
[layer_regularization_losses
!regularization_losses
 
 
 
�

\layers
#trainable_variables
]non_trainable_variables
$	variables
^metrics
_layer_metrics
`layer_regularization_losses
%regularization_losses
 
 
 
�

alayers
'trainable_variables
bnon_trainable_variables
(	variables
cmetrics
dlayer_metrics
elayer_regularization_losses
)regularization_losses
 
 
 
�

flayers
+trainable_variables
gnon_trainable_variables
,	variables
hmetrics
ilayer_metrics
jlayer_regularization_losses
-regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
�

klayers
1trainable_variables
lnon_trainable_variables
2	variables
mmetrics
nlayer_metrics
olayer_regularization_losses
3regularization_losses
 
 
 
�

players
5trainable_variables
qnon_trainable_variables
6	variables
rmetrics
slayer_metrics
tlayer_regularization_losses
7regularization_losses
 
 
 
�

ulayers
9trainable_variables
vnon_trainable_variables
:	variables
wmetrics
xlayer_metrics
ylayer_regularization_losses
;regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
�

zlayers
?trainable_variables
{non_trainable_variables
@	variables
|metrics
}layer_metrics
~layer_regularization_losses
Aregularization_losses
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
^
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
9
10
11
12
 

0
�1
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
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_2/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_2/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_3Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3embedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*,
f'R%
#__inference_signature_wrapper_22324
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*'
f"R 
__inference__traced_save_23082
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/embedding_1/embeddings/mAdam/embedding_2/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/embedding_1/embeddings/vAdam/embedding_2/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v**
Tin#
!2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8**
f%R#
!__inference__traced_restore_23184��
�
�
%__inference_model_layer_call_fn_22559
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_221132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
l
2__inference_tf_op_layer_concat_layer_call_fn_22694
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_216972
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:U Q
+
_output_shapes
:���������2
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/2
�
f
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_22789

inputs
identityS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yj
PowPowinputsPow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
Pow[
IdentityIdentityPow:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
K
/__inference_tf_op_layer_Pow_layer_call_fn_22794

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_217992
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
%__inference_dense_layer_call_fn_22783

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_217772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�z
�
@__inference_model_layer_call_and_return_conditional_losses_22113

inputs
inputs_1
inputs_2
embedding_22032
embedding_1_22035
embedding_2_22038
dense_22044
dense_22046
dense_1_22051
dense_1_22053
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_22032*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_216172#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_22035*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_216472%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_22038*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_216772%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_216972$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_217132%
#tf_op_layer_Reshape/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217332!
dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_22044dense_22046*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_217772
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_217992!
tf_op_layer_Pow/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218192#
!dropout_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_22051dense_1_22053*
Tin
2*
Tout
2*'
_output_shapes
:���������J*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_218632!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_22032* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_22035*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_22038*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22044*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_22046*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_22051*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_22053*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: 
�
O
3__inference_tf_op_layer_Reshape_layer_call_fn_22705

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_217132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_22774

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�5
�
 __inference__wrapped_model_21593
input_1
input_2
input_3*
&model_embedding_embedding_lookup_21555,
(model_embedding_1_embedding_lookup_21561,
(model_embedding_2_embedding_lookup_21567.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity�~
model/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding/Cast�
 model/embedding/embedding_lookupResourceGather&model_embedding_embedding_lookup_21555model/embedding/Cast:y:0*
Tindices0*9
_class/
-+loc:@model/embedding/embedding_lookup/21555*+
_output_shapes
:���������2*
dtype02"
 model/embedding/embedding_lookup�
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@model/embedding/embedding_lookup/21555*+
_output_shapes
:���������22+
)model/embedding/embedding_lookup/Identity�
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22-
+model/embedding/embedding_lookup/Identity_1�
model/embedding_1/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding_1/Cast�
"model/embedding_1/embedding_lookupResourceGather(model_embedding_1_embedding_lookup_21561model/embedding_1/Cast:y:0*
Tindices0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/21561*+
_output_shapes
:���������2*
dtype02$
"model/embedding_1/embedding_lookup�
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/21561*+
_output_shapes
:���������22-
+model/embedding_1/embedding_lookup/Identity�
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22/
-model/embedding_1/embedding_lookup/Identity_1�
model/embedding_2/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding_2/Cast�
"model/embedding_2/embedding_lookupResourceGather(model_embedding_2_embedding_lookup_21567model/embedding_2/Cast:y:0*
Tindices0*;
_class1
/-loc:@model/embedding_2/embedding_lookup/21567*+
_output_shapes
:���������2*
dtype02$
"model/embedding_2/embedding_lookup�
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_2/embedding_lookup/21567*+
_output_shapes
:���������22-
+model/embedding_2/embedding_lookup/Identity�
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22/
-model/embedding_2/embedding_lookup/Identity_1�
$model/tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/tf_op_layer_concat/concat/axis�
model/tf_op_layer_concat/concatConcatV24model/embedding/embedding_lookup/Identity_1:output:06model/embedding_1/embedding_lookup/Identity_1:output:06model/embedding_2/embedding_lookup/Identity_1:output:0-model/tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022!
model/tf_op_layer_concat/concat�
'model/tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2)
'model/tf_op_layer_Reshape/Reshape/shape�
!model/tf_op_layer_Reshape/ReshapeReshape(model/tf_op_layer_concat/concat:output:00model/tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2#
!model/tf_op_layer_Reshape/Reshape�
model/dropout/IdentityIdentity*model/tf_op_layer_Reshape/Reshape:output:0*
T0*(
_output_shapes
:����������2
model/dropout/Identity�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulmodel/dropout/Identity:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/dense/BiasAdd
model/tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
model/tf_op_layer_Pow/Pow/y�
model/tf_op_layer_Pow/PowPowmodel/dense/BiasAdd:output:0$model/tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
model/tf_op_layer_Pow/Pow�
model/dropout_1/IdentityIdentitymodel/tf_op_layer_Pow/Pow:z:0*
T0*'
_output_shapes
:���������d2
model/dropout_1/Identity�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:dJ*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
model/dense_1/BiasAddr
IdentityIdentitymodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
%__inference_model_layer_call_fn_22237
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_222202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
E
)__inference_dropout_1_layer_call_fn_22821

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
o
)__inference_embedding_layer_call_fn_22613

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_216172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
f
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_21799

inputs
identityS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yj
PowPowinputsPow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
Pow[
IdentityIdentityPow:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
q
+__inference_embedding_2_layer_call_fn_22679

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_216772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
#__inference_signature_wrapper_22324
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*)
f$R"
 __inference__wrapped_model_215932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_22687
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:U Q
+
_output_shapes
:���������2
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/2
�
m
__inference_loss_fn_5_22950=
9dense_1_kernel_regularizer_square_readvariableop_resource
identity��
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/adde
IdentityIdentity"dense_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�x
�
@__inference_model_layer_call_and_return_conditional_losses_22022
input_1
input_2
input_3
embedding_21941
embedding_1_21944
embedding_2_21947
dense_21953
dense_21955
dense_1_21960
dense_1_21962
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_21941*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_216172#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_21944*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_216472%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_21947*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_216772%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_216972$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_217132%
#tf_op_layer_Reshape/PartitionedCall�
dropout/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217382
dropout/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_21953dense_21955*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_217772
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_217992!
tf_op_layer_Pow/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218242
dropout_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_21960dense_1_21962*
Tin
2*
Tout
2*'
_output_shapes
:���������J*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_218632!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_21941* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_21944*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_21947*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21953*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_21955*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_21960*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_21962*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�z
�
@__inference_model_layer_call_and_return_conditional_losses_21936
input_1
input_2
input_3
embedding_21626
embedding_1_21656
embedding_2_21686
dense_21788
dense_21790
dense_1_21874
dense_1_21876
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_21626*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_216172#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_21656*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_216472%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_21686*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_216772%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_216972$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_217132%
#tf_op_layer_Reshape/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217332!
dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_21788dense_21790*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_217772
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_217992!
tf_op_layer_Pow/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218192#
!dropout_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_21874dense_1_21876*
Tin
2*
Tout
2*'
_output_shapes
:���������J*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_218632!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_21626* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_21656*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_21686*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21788*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_21790*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_21874*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_21876*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
F__inference_embedding_2_layer_call_and_return_conditional_losses_21677

inputs
embedding_lookup_21663
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_21663Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/21663*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21663*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_21663*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
j
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_22700

inputs
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs
�
k
__inference_loss_fn_6_22963;
7dense_1_bias_regularizer_square_readvariableop_resource
identity��
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_square_readvariableop_resource*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addc
IdentityIdentity dense_1/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_21647

inputs
embedding_lookup_21633
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_21633Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/21633*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21633*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_21633*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
F__inference_embedding_2_layer_call_and_return_conditional_losses_22672

inputs
embedding_lookup_22658
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_22658Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22658*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22658*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22658*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
C
'__inference_dropout_layer_call_fn_22732

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
u
__inference_loss_fn_1_22898E
Aembedding_1_embeddings_regularizer_square_readvariableop_resource
identity��
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_1_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addm
IdentityIdentity*embedding_1/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�

D__inference_embedding_layer_call_and_return_conditional_losses_22606

inputs
embedding_lookup_22592
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_22592Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22592*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22592*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22592* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
b
)__inference_dropout_1_layer_call_fn_22816

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
q
+__inference_embedding_1_layer_call_fn_22646

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_216472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
|
'__inference_dense_1_layer_call_fn_22872

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������J*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_218632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
`
'__inference_dropout_layer_call_fn_22727

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
__inference_loss_fn_4_229379
5dense_bias_regularizer_square_readvariableop_resource
identity��
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_bias_regularizer_square_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/adda
IdentityIdentitydense/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
͑
�
@__inference_model_layer_call_and_return_conditional_losses_22438
inputs_0
inputs_1
inputs_2$
 embedding_embedding_lookup_22330&
"embedding_1_embedding_lookup_22336&
"embedding_2_embedding_lookup_22342(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGather embedding_embedding_lookup_22330embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/22330*+
_output_shapes
:���������2*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/22330*+
_output_shapes
:���������22%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22'
%embedding/embedding_lookup/Identity_1w
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_22336embedding_1/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/22336*+
_output_shapes
:���������2*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/22336*+
_output_shapes
:���������22'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_1/embedding_lookup/Identity_1w
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_2/Cast�
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_22342embedding_2/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/22342*+
_output_shapes
:���������2*
dtype02
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/22342*+
_output_shapes
:���������22'
%embedding_2/embedding_lookup/Identity�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_2/embedding_lookup/Identity_1�
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
tf_op_layer_concat/concat/axis�
tf_op_layer_concat/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
tf_op_layer_concat/concat�
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2#
!tf_op_layer_Reshape/Reshape/shape�
tf_op_layer_Reshape/ReshapeReshape"tf_op_layer_concat/concat:output:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Reshape/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/dropout/Const�
dropout/dropout/MulMul$tf_op_layer_Reshape/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mul�
dropout/dropout/ShapeShape$tf_op_layer_Reshape/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mul_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAdds
tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf_op_layer_Pow/Pow/y�
tf_op_layer_Pow/PowPowdense/BiasAdd:output:0tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
tf_op_layer_Pow/Poww
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMultf_op_layer_Pow/Pow:z:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout_1/dropout/Muly
dropout_1/dropout/ShapeShapetf_op_layer_Pow/Pow:z:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout_1/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dJ*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
dense_1/BiasAdd�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp embedding_embedding_lookup_22330* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_1_embedding_lookup_22336*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_2_embedding_lookup_22342*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
a
B__inference_dropout_layer_call_and_return_conditional_losses_21733

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
!__inference__traced_restore_23184
file_prefix)
%assignvariableop_embedding_embeddings-
)assignvariableop_1_embedding_1_embeddings-
)assignvariableop_2_embedding_2_embeddings#
assignvariableop_3_dense_kernel!
assignvariableop_4_dense_bias%
!assignvariableop_5_dense_1_kernel#
assignvariableop_6_dense_1_bias 
assignvariableop_7_adam_iter"
assignvariableop_8_adam_beta_1"
assignvariableop_9_adam_beta_2"
assignvariableop_10_adam_decay*
&assignvariableop_11_adam_learning_rate
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_13
/assignvariableop_16_adam_embedding_embeddings_m5
1assignvariableop_17_adam_embedding_1_embeddings_m5
1assignvariableop_18_adam_embedding_2_embeddings_m+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m-
)assignvariableop_21_adam_dense_1_kernel_m+
'assignvariableop_22_adam_dense_1_bias_m3
/assignvariableop_23_adam_embedding_embeddings_v5
1assignvariableop_24_adam_embedding_1_embeddings_v5
1assignvariableop_25_adam_embedding_2_embeddings_v+
'assignvariableop_26_adam_dense_kernel_v)
%assignvariableop_27_adam_dense_bias_v-
)assignvariableop_28_adam_dense_1_kernel_v+
'assignvariableop_29_adam_dense_1_bias_v
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_2_embeddingsIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0	*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_embedding_1_embeddings_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_embedding_2_embeddings_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_embedding_1_embeddings_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_2_embeddings_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30�
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*�
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
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
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_21819

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_21697

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������2
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_22538
inputs_0
inputs_1
inputs_2$
 embedding_embedding_lookup_22444&
"embedding_1_embedding_lookup_22450&
"embedding_2_embedding_lookup_22456(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGather embedding_embedding_lookup_22444embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/22444*+
_output_shapes
:���������2*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/22444*+
_output_shapes
:���������22%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22'
%embedding/embedding_lookup/Identity_1w
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_22450embedding_1/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/22450*+
_output_shapes
:���������2*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/22450*+
_output_shapes
:���������22'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_1/embedding_lookup/Identity_1w
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_2/Cast�
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_22456embedding_2/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/22456*+
_output_shapes
:���������2*
dtype02
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/22456*+
_output_shapes
:���������22'
%embedding_2/embedding_lookup/Identity�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_2/embedding_lookup/Identity_1�
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
tf_op_layer_concat/concat/axis�
tf_op_layer_concat/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
tf_op_layer_concat/concat�
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2#
!tf_op_layer_Reshape/Reshape/shape�
tf_op_layer_Reshape/ReshapeReshape"tf_op_layer_concat/concat:output:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Reshape/Reshape�
dropout/IdentityIdentity$tf_op_layer_Reshape/Reshape:output:0*
T0*(
_output_shapes
:����������2
dropout/Identity�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAdds
tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf_op_layer_Pow/Pow/y�
tf_op_layer_Pow/PowPowdense/BiasAdd:output:0tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
tf_op_layer_Pow/Pow
dropout_1/IdentityIdentitytf_op_layer_Pow/Pow:z:0*
T0*'
_output_shapes
:���������d2
dropout_1/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dJ*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
dense_1/BiasAdd�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp embedding_embedding_lookup_22444* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_1_embedding_lookup_22450*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_2_embedding_lookup_22456*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
%__inference_model_layer_call_fn_22130
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_221132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_21738

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_21863

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dJ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2	
BiasAdd�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�K
�
__inference__traced_save_23082
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8792053be7d04af0a4377f4895a530b1/part2	
Const_1�
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��2:02:I2:	�d:d:dJ:J: : : : : : : : : :
��2:02:I2:	�d:d:dJ:J:
��2:02:I2:	�d:d:dJ:J: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��2:$ 

_output_shapes

:02:$ 

_output_shapes

:I2:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dJ: 

_output_shapes
:J:

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :&"
 
_output_shapes
:
��2:$ 

_output_shapes

:02:$ 

_output_shapes

:I2:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dJ: 

_output_shapes
:J:&"
 
_output_shapes
:
��2:$ 

_output_shapes

:02:$ 

_output_shapes

:I2:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dJ: 

_output_shapes
:J:

_output_shapes
: 
�x
�
@__inference_model_layer_call_and_return_conditional_losses_22220

inputs
inputs_1
inputs_2
embedding_22139
embedding_1_22142
embedding_2_22145
dense_22151
dense_22153
dense_1_22158
dense_1_22160
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_22139*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_216172#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_22142*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_216472%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_22145*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_216772%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_216972$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*W
fRRP
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_217132%
#tf_op_layer_Reshape/PartitionedCall�
dropout/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_217382
dropout/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_22151dense_22153*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_217772
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*S
fNRL
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_217992!
tf_op_layer_Pow/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

CPU

GPU2 *0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_218242
dropout_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_22158dense_1_22160*
Tin
2*
Tout
2*'
_output_shapes
:���������J*$
_read_only_resource_inputs
*/
config_proto

CPU

GPU2 *0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_218632!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_22139* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_22142*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_22145*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22151*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_22153*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_22158*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_22160*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: 
�

D__inference_embedding_layer_call_and_return_conditional_losses_21617

inputs
embedding_lookup_21603
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_21603Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/21603*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21603*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_21603* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_21824

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_22580
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*'
_output_shapes
:���������J*)
_read_only_resource_inputs
		*/
config_proto

CPU

GPU2 *0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_222202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_22639

inputs
embedding_lookup_22625
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_22625Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22625*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22625*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22625*
_output_shapes

:02*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:022+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_22811

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_22806

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
u
__inference_loss_fn_2_22911E
Aembedding_2_embeddings_regularizer_square_readvariableop_resource
identity��
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:I2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:I22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addm
IdentityIdentity*embedding_2/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_22722

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
__inference_loss_fn_3_22924;
7dense_kernel_regularizer_square_readvariableop_resource
identity��
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addc
IdentityIdentity dense/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_22863

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dJ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J2	
BiasAdd�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dJ*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dJ2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:J2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������J2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense_layer_call_and_return_conditional_losses_21777

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
a
B__inference_dropout_layer_call_and_return_conditional_losses_22717

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
__inference_loss_fn_0_22885C
?embedding_embeddings_regularizer_square_readvariableop_resource
identity��
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp?embedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
��2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addk
IdentityIdentity(embedding/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
j
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_21713

inputs
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������
;
input_20
serving_default_input_2:0���������
;
input_30
serving_default_input_3:0���������;
dense_10
StatefulPartitionedCall:0���������Jtensorflow/serving/predict:��
�b
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�^
_tf_keras_model�^{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 21679, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 48, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 73, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "3"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"3": 1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 2400]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}, "name": "tf_op_layer_Pow", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["tf_op_layer_Pow", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 74, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 12]}]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 21679, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 48, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 73, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "3"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"3": 1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 2400]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}, "name": "tf_op_layer_Pow", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["tf_op_layer_Pow", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 74, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 21679, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 48, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
�

embeddings
trainable_variables
 	variables
!regularization_losses
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 73, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
�
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "3"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"3": 1}}}
�
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 2400]}}}
�
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2400]}}
�
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}}
�
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 74, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m�/m�0m�=m�>m�v�v�v�/v�0v�=v�>v�"
	optimizer
Q
0
1
2
/3
04
=5
>6"
trackable_list_wrapper
Q
0
1
2
/3
04
=5
>6"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
�

Hlayers
trainable_variables
Inon_trainable_variables
	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
(:&
��22embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�

Mlayers
trainable_variables
Nnon_trainable_variables
	variables
Ometrics
Player_metrics
Qlayer_regularization_losses
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&022embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�

Rlayers
trainable_variables
Snon_trainable_variables
	variables
Tmetrics
Ulayer_metrics
Vlayer_regularization_losses
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&I22embedding_2/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�

Wlayers
trainable_variables
Xnon_trainable_variables
 	variables
Ymetrics
Zlayer_metrics
[layer_regularization_losses
!regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

\layers
#trainable_variables
]non_trainable_variables
$	variables
^metrics
_layer_metrics
`layer_regularization_losses
%regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

alayers
'trainable_variables
bnon_trainable_variables
(	variables
cmetrics
dlayer_metrics
elayer_regularization_losses
)regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

flayers
+trainable_variables
gnon_trainable_variables
,	variables
hmetrics
ilayer_metrics
jlayer_regularization_losses
-regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�d2dense/kernel
:d2
dense/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�

klayers
1trainable_variables
lnon_trainable_variables
2	variables
mmetrics
nlayer_metrics
olayer_regularization_losses
3regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

players
5trainable_variables
qnon_trainable_variables
6	variables
rmetrics
slayer_metrics
tlayer_regularization_losses
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

ulayers
9trainable_variables
vnon_trainable_variables
:	variables
wmetrics
xlayer_metrics
ylayer_regularization_losses
;regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :dJ2dense_1/kernel
:J2dense_1/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�

zlayers
?trainable_variables
{non_trainable_variables
@	variables
|metrics
}layer_metrics
~layer_regularization_losses
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
~
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
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
�1"
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
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
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
-:+
��22Adam/embedding/embeddings/m
-:+022Adam/embedding_1/embeddings/m
-:+I22Adam/embedding_2/embeddings/m
$:"	�d2Adam/dense/kernel/m
:d2Adam/dense/bias/m
%:#dJ2Adam/dense_1/kernel/m
:J2Adam/dense_1/bias/m
-:+
��22Adam/embedding/embeddings/v
-:+022Adam/embedding_1/embeddings/v
-:+I22Adam/embedding_2/embeddings/v
$:"	�d2Adam/dense/kernel/v
:d2Adam/dense/bias/v
%:#dJ2Adam/dense_1/kernel/v
:J2Adam/dense_1/bias/v
�2�
@__inference_model_layer_call_and_return_conditional_losses_22022
@__inference_model_layer_call_and_return_conditional_losses_21936
@__inference_model_layer_call_and_return_conditional_losses_22438
@__inference_model_layer_call_and_return_conditional_losses_22538�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_model_layer_call_fn_22559
%__inference_model_layer_call_fn_22237
%__inference_model_layer_call_fn_22130
%__inference_model_layer_call_fn_22580�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_21593�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *q�n
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
�2�
D__inference_embedding_layer_call_and_return_conditional_losses_22606�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_embedding_layer_call_fn_22613�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_embedding_1_layer_call_and_return_conditional_losses_22639�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_embedding_1_layer_call_fn_22646�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_embedding_2_layer_call_and_return_conditional_losses_22672�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_embedding_2_layer_call_fn_22679�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_22687�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_tf_op_layer_concat_layer_call_fn_22694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_22700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_tf_op_layer_Reshape_layer_call_fn_22705�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dropout_layer_call_and_return_conditional_losses_22717
B__inference_dropout_layer_call_and_return_conditional_losses_22722�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dropout_layer_call_fn_22732
'__inference_dropout_layer_call_fn_22727�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_22774�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_22783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_22789�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_tf_op_layer_Pow_layer_call_fn_22794�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dropout_1_layer_call_and_return_conditional_losses_22811
D__inference_dropout_1_layer_call_and_return_conditional_losses_22806�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_1_layer_call_fn_22821
)__inference_dropout_1_layer_call_fn_22816�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_22863�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_22872�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_22885�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_22898�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_22911�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_22924�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_22937�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_22950�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_6_22963�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
BB@
#__inference_signature_wrapper_22324input_1input_2input_3�
 __inference__wrapped_model_21593�/0=>{�x
q�n
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
� "1�.
,
dense_1!�
dense_1���������J�
B__inference_dense_1_layer_call_and_return_conditional_losses_22863\=>/�,
%�"
 �
inputs���������d
� "%�"
�
0���������J
� z
'__inference_dense_1_layer_call_fn_22872O=>/�,
%�"
 �
inputs���������d
� "����������J�
@__inference_dense_layer_call_and_return_conditional_losses_22774]/00�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� y
%__inference_dense_layer_call_fn_22783P/00�-
&�#
!�
inputs����������
� "����������d�
D__inference_dropout_1_layer_call_and_return_conditional_losses_22806\3�0
)�&
 �
inputs���������d
p
� "%�"
�
0���������d
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_22811\3�0
)�&
 �
inputs���������d
p 
� "%�"
�
0���������d
� |
)__inference_dropout_1_layer_call_fn_22816O3�0
)�&
 �
inputs���������d
p
� "����������d|
)__inference_dropout_1_layer_call_fn_22821O3�0
)�&
 �
inputs���������d
p 
� "����������d�
B__inference_dropout_layer_call_and_return_conditional_losses_22717^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_22722^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� |
'__inference_dropout_layer_call_fn_22727Q4�1
*�'
!�
inputs����������
p
� "�����������|
'__inference_dropout_layer_call_fn_22732Q4�1
*�'
!�
inputs����������
p 
� "������������
F__inference_embedding_1_layer_call_and_return_conditional_losses_22639_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� �
+__inference_embedding_1_layer_call_fn_22646R/�,
%�"
 �
inputs���������
� "����������2�
F__inference_embedding_2_layer_call_and_return_conditional_losses_22672_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� �
+__inference_embedding_2_layer_call_fn_22679R/�,
%�"
 �
inputs���������
� "����������2�
D__inference_embedding_layer_call_and_return_conditional_losses_22606_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� 
)__inference_embedding_layer_call_fn_22613R/�,
%�"
 �
inputs���������
� "����������2:
__inference_loss_fn_0_22885�

� 
� "� :
__inference_loss_fn_1_22898�

� 
� "� :
__inference_loss_fn_2_22911�

� 
� "� :
__inference_loss_fn_3_22924/�

� 
� "� :
__inference_loss_fn_4_229370�

� 
� "� :
__inference_loss_fn_5_22950=�

� 
� "� :
__inference_loss_fn_6_22963>�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_21936�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p

 
� "%�"
�
0���������J
� �
@__inference_model_layer_call_and_return_conditional_losses_22022�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p 

 
� "%�"
�
0���������J
� �
@__inference_model_layer_call_and_return_conditional_losses_22438�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "%�"
�
0���������J
� �
@__inference_model_layer_call_and_return_conditional_losses_22538�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "%�"
�
0���������J
� �
%__inference_model_layer_call_fn_22130�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p

 
� "����������J�
%__inference_model_layer_call_fn_22237�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p 

 
� "����������J�
%__inference_model_layer_call_fn_22559�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "����������J�
%__inference_model_layer_call_fn_22580�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "����������J�
#__inference_signature_wrapper_22324�/0=>���
� 
���
,
input_1!�
input_1���������
,
input_2!�
input_2���������
,
input_3!�
input_3���������"1�.
,
dense_1!�
dense_1���������J�
J__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_22789X/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
/__inference_tf_op_layer_Pow_layer_call_fn_22794K/�,
%�"
 �
inputs���������d
� "����������d�
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_22700]3�0
)�&
$�!
inputs���������02
� "&�#
�
0����������
� �
3__inference_tf_op_layer_Reshape_layer_call_fn_22705P3�0
)�&
$�!
inputs���������02
� "������������
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_22687����
��}
{�x
&�#
inputs/0���������2
&�#
inputs/1���������2
&�#
inputs/2���������2
� ")�&
�
0���������02
� �
2__inference_tf_op_layer_concat_layer_call_fn_22694����
��}
{�x
&�#
inputs/0���������2
&�#
inputs/1���������2
&�#
inputs/2���������2
� "����������02