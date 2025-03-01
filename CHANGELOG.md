# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-03-01

- Revised implementation to only substantive changes to base pykan code.  Uses dynamic override of __getattribute__() to redirect calls to MultKAN / KANLayer to MatrixKAN / MatrixKANLayer.  Usage is unchanged.

## [1.0.0] - 2025-02-12

- Initial implementation