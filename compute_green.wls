#!/usr/bin/env wolframscript
(* ::Package:: *)

args = $ScriptCommandLine;
ToExpression/@Rest[args]
(*Expect: maxOrder eps*)
Print["Max order = "<>ToString@maxOrder]

R={x,y,z};
u={0,0,1};
RcrossU=R\[Cross]u;
RcrossU2=RcrossU . RcrossU;
\[Epsilon]=DiagonalMatrix[{1,1,\[Epsilon]zz}];
\[Epsilon]u=\[Epsilon]0 \[Epsilon]zz;
\[Epsilon]uni=\[Epsilon]0 \[Epsilon];
r\[Epsilon]=\[Epsilon]u R . Inverse[\[Epsilon]uni] . R//Simplify;
g\[Epsilon]uni=E^(I k0 Sqrt[r\[Epsilon]])/(4\[Pi] Sqrt[r\[Epsilon]])//Simplify;
r\[Mu]=R . R//Simplify;
g\[Mu]uni=E^(I k0 Sqrt[r\[Mu]])/(4\[Pi] Sqrt[r\[Mu]])//Simplify;
DelDel[f_]:=Table[D[f,R[[i]],R[[j]]],{i,1,3},{j,1,3}];
B=RcrossU\[TensorProduct]RcrossU/RcrossU2(\[Epsilon]u/\[Epsilon]0 g\[Epsilon]uni-g\[Mu]uni)+1/(I k0 RcrossU2)(IdentityMatrix[3]-u\[TensorProduct]u-2RcrossU\[TensorProduct]RcrossU/RcrossU2)(g\[Epsilon]uni Sqrt[r\[Epsilon]]-g\[Mu]uni Sqrt[r\[Mu]]);
GChen=-B+\[Epsilon]u Inverse[\[Epsilon]uni]g\[Epsilon]uni+1/(k0^2)DelDel[g\[Epsilon]uni];
rn=1;
sampleNumber=51;
\[Epsilon]zzn=eps;

Print["eps = "<>ToString@\[Epsilon]zzn]

margin=8/Sqrt@\[Epsilon]zzn;
\[Phi]n=Subdivide[Clip[\[Pi]/2-margin,{0,2\[Pi]}],Clip[\[Pi]/2+margin,{0,2\[Pi]}],sampleNumber];
\[Theta]n=Subdivide[Clip[\[Pi]/2-margin/2,{0,\[Pi]}],Clip[\[Pi]/2+margin/2,{0,\[Pi]}],Round[sampleNumber/2]];
k0n=2\[Pi];
numeralize[g_]:=Parallelize@Outer[
Function[{\[Theta],\[Phi]},
N[g/.{y->rn Cos[\[Phi]]Sin[\[Theta]],z->rn Sin[\[Phi]]Sin[\[Theta]],x->rn Cos[\[Theta]],\[Epsilon]zz->\[Epsilon]zzn,k0->k0n}]
],
\[Theta]n,\[Phi]n
];

dimKept=1;
GChen[[;;,2]]=0;
GChen[[;;,3]]=0;

cartesians[ax_,ay_,az_]:=cartesians[ax,ay,az]=If[ax>0,
D[cartesians[ax-1,ay,az],x],
If[ay>0,
D[cartesians[ax,ay-1,az],y],
If[az>0,
D[cartesians[ax,ay,az-1],z],
GChen
]]]
Table[cartesians[ax,ay,az],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];
ToMultiIndex[coefficient_,powers_]:=If[powers=={},
{},
Total@(UnitVector[3,#]&/@powers)->coefficient]
CoefficientArraysToMultiIndex[array_]:=If[Length@array==1,
<|{0,0,0}->array[[1]]|>,
Association@Select[Flatten@Select[MapIndexed[ToMultiIndex,#,{Length@Dimensions@#}]&/@array,Length@#>0&]/.(index_->0):>{},Length@#>0&]];
dr=dx Sin[\[Theta]]Cos[\[Phi]]+dy Sin[\[Theta]]Sin[\[Phi]]+dz Cos[\[Theta]];
sphericalToCartesian=Table[{l,m}->4\[Pi](-I)(-1)^l/k0n^(l+1)
CoefficientArraysToMultiIndex[CoefficientArrays[(dx^2+dy^2+dz^2)^(l/2)Simplify@TrigExpand@ExpToTrig@TransformedField[
"Spherical"->"Cartesian",
SphericalHarmonicY[l,m,\[Theta],\[Phi]],
{\[Rho],\[Theta],\[Phi]}->{dx,dy,dz}
],{dx,dy,dz}]],
{l,0,maxOrder},{m,-l,l}]//Association;
cartesiansN=Association@Flatten@Table[Print[ax,ay,az];{ax,ay,az}->numeralize[cartesians[ax,ay,az]],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];
sphericalsN=Association@Flatten@Table[{l,m}->Total@KeyValueMap[{\[Alpha],value}|->value cartesiansN[\[Alpha]],sphericalToCartesian[{l,m}]],
{l,0,maxOrder},{m,-l,l}];
saveToFile[\[Alpha]_]:=Module[{filename,fd,toExport},
filename=FileNameJoin[{Directory[],"data","mathematica","eps_"<>ToString@\[Epsilon]zzn<>"/"<>ToString@\[Alpha][[1]]<>ToString@\[Alpha][[2]]<>ToString@\[Alpha][[3]]<>".txt"}];
fd=OpenWrite[filename];
Table[WriteString[filename,"This line empty on purpose\n"],{i,1,9}];
Close[fd];
fd=OpenAppend[filename];
toExport=cartesiansN[\[Alpha]];
Table[
WriteString[filename,
ToString[N[rn Cos[\[Theta]n[[i]]]],CForm]<>" "<>
ToString[N[rn Cos[\[Phi]n[[j]]]Sin[\[Theta]n[[i]]]],CForm]<>" "<>
ToString[N[rn Sin[\[Phi]n[[j]]]Sin[\[Theta]n[[i]]]],CForm]<>" "<>
ToString[N[Re@toExport[[i,j,1,1]]],CForm]<>" "<>
ToString[N[Im@toExport[[i,j,1,1]]],CForm]<>" "<>
ToString[N[Re@toExport[[i,j,2,1]]],CForm]<>" "<>
ToString[N[Im@toExport[[i,j,2,1]]],CForm]<>" "<>
ToString[N[Re@toExport[[i,j,3,1]]],CForm]<>" "<>
ToString[N[Im@toExport[[i,j,3,1]]],CForm]<>" "<>
"\n"],
{i,1,Length@\[Theta]n},{j,1,Length@\[Phi]n}];
Close@fd;
]
Table[saveToFile[{ax,ay,az}],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];
