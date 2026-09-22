// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_hand_v0_generated.h>

#include <cstddef>
#include <memory>
#include <type_traits>

#define VT(field) (field + 2) * 2
static_assert(core::SomaHandPoseV0::VT_JOINT_ROTATIONS == VT(0));
static_assert(core::SomaHandPoseV0::VT_GLOBAL_TRANSLATION == VT(1));
static_assert(core::SomaHandPoseV0::VT_GLOBAL_TRANSLATION_IS_VALID == VT(2));
static_assert(core::SomaHandPoseV0::VT_HANDEDNESS == VT(3));
static_assert(core::SomaHandPoseV0Record::VT_DATA == VT(0));
static_assert(core::SomaHandPoseV0Record::VT_TIMESTAMP == VT(1));

#define TYPE(field) decltype(std::declval<core::SomaHandPoseV0>().field())
static_assert(std::is_same_v<TYPE(joint_rotations), const core::SomaHandJointRotationsV0*>);
static_assert(std::is_same_v<TYPE(global_translation), const core::Point*>);
static_assert(std::is_same_v<TYPE(global_translation_is_valid), bool>);
static_assert(std::is_same_v<TYPE(handedness), core::SomaHandednessV0>);

static_assert(std::is_trivially_copyable_v<core::SomaJointRotationV0>);
static_assert(std::is_trivially_copyable_v<core::SomaHandJointRotationsV0>);
static_assert(sizeof(core::SomaHandJointRotationsV0) == 25 * sizeof(core::SomaJointRotationV0));

static_assert(core::SomaHandJointV0_WRIST == 0);
static_assert(core::SomaHandJointV0_INDEX1 == 5);
static_assert(core::SomaHandJointV0_MIDDLE1 == 10);
static_assert(core::SomaHandJointV0_RING1 == 15);
static_assert(core::SomaHandJointV0_PINKY_END == 24);
static_assert(core::SomaHandJointV0_NUM_JOINTS == 25);

TEST_CASE("SOMA hand v0 rotation array has the public layout", "[soma_hand_v0][struct]")
{
    core::SomaHandJointRotationsV0 rotations;

    REQUIRE(rotations.values()->size() == static_cast<size_t>(core::SomaHandJointV0_NUM_JOINTS));
    CHECK((*rotations.values())[core::SomaHandJointV0_WRIST]->axis_angle().x() == 0.0f);
    CHECK_FALSE((*rotations.values())[core::SomaHandJointV0_PINKY_END]->is_valid());

    const core::SomaJointRotationV0 wrist(core::Point(0.1f, 0.2f, 0.3f), true);
    rotations.mutable_values()->Mutate(core::SomaHandJointV0_WRIST, wrist);

    CHECK((*rotations.values())[core::SomaHandJointV0_WRIST]->axis_angle().y() == Catch::Approx(0.2f));
    CHECK((*rotations.values())[core::SomaHandJointV0_WRIST]->is_valid());
}

TEST_CASE("SOMA hand pose v0 round trips through FlatBuffers", "[soma_hand_v0][flatbuffers]")
{
    core::SomaHandPoseV0T pose;
    pose.joint_rotations = std::make_shared<core::SomaHandJointRotationsV0>();
    pose.joint_rotations->mutable_values()->Mutate(
        core::SomaHandJointV0_INDEX_END, core::SomaJointRotationV0(core::Point(0.4f, 0.5f, 0.6f), true));
    pose.global_translation = std::make_shared<core::Point>(1.0f, 2.0f, 3.0f);
    pose.global_translation_is_valid = true;
    pose.handedness = core::SomaHandednessV0_RIGHT;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaHandPoseV0::Pack(builder, &pose));

    const auto* decoded = flatbuffers::GetRoot<core::SomaHandPoseV0>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_rotations() != nullptr);
    REQUIRE(decoded->global_translation() != nullptr);
    const auto* index_end = (*decoded->joint_rotations()->values())[core::SomaHandJointV0_INDEX_END];
    CHECK(index_end->axis_angle().z() == Catch::Approx(0.6f));
    CHECK(index_end->is_valid());
    CHECK(decoded->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->global_translation_is_valid());
    CHECK(decoded->handedness() == core::SomaHandednessV0_RIGHT);
}
